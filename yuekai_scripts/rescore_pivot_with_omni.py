# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Re-profile pivot difficulty with the target policy (PivotRL-style).

For every row of a pivot JSONL, samples K rollouts from the given model at the
training temperature (prompts rendered with the recipe's inlined chat template)
and scores each rollout with a faithful local copy of the Gym
single_step_tool_use_with_argument_comparison verifier. Emits one compact JSON
line per row: {"i", "expected_type", "rewards", "reward_mean", "reward_var",
"n_truncated"}. Supports sharding (--shard k/n) and resume (skips rows already
present in the output file).

Example:
  python yuekai_scripts/rescore_pivot_with_omni.py \
    --model Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --in-jsonl datasets/conversational_tool_use_pivot/train.jsonl \
    --recipe-yaml examples/nemo_gym/grpo_qwen3_omni_30ba3b_thinking_tau1.yaml \
    --out results/omni_rescore/shard0.jsonl --shard 0/2 --samples 8
"""

import argparse
import json
import re
from collections import Counter

import jinja2
import yaml

WORD_SIM_THRESHOLD = 0.1
FLOAT_THRESHOLD = 1e-6


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--in-jsonl", required=True)
    p.add_argument("--recipe-yaml", default=None, help="Required for --template-mode recipe")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--template-mode",
        choices=("recipe", "tokenizer"),
        default="recipe",
        help="recipe: inlined chat_template from the recipe yaml (qwen3-omni); "
        "tokenizer: AutoTokenizer.apply_chat_template with enable_thinking=True (nano-omni)",
    )
    p.add_argument(
        "--tool-format",
        choices=("hermes", "qwen3_coder"),
        default="hermes",
        help="hermes: <tool_call>{json}</tool_call>; qwen3_coder: <function=name><parameter=k>v</parameter>",
    )
    p.add_argument("--bad-words", default="", help="comma-separated bad words for sampling")
    p.add_argument("--no-enforce-eager", action="store_true", help="enable CUDA graphs (mamba models)")
    p.add_argument("--mamba-f32-cache", action="store_true")
    p.add_argument("--shard", default="0/1", help="k/n: process rows with index %% n == k")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--chunk", type=int, default=384, help="prompts per generate/flush cycle")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument(
        "--max-prompt-chars",
        type=int,
        default=85000,
        help="Rows with longer rendered prompts are skipped with a placeholder line "
        "(a single oversized prompt otherwise fails the whole generate batch).",
    )
    return p.parse_args()


def compare_arguments(expected, actual) -> bool:
    """Faithful copy of Gym ToolCallComparator.compare_tool_call_arguments."""
    if not isinstance(actual, type(expected)):
        return False
    if isinstance(expected, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(compare_arguments(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False
        return all(compare_arguments(e, a) for e, a in zip(expected, actual))
    if isinstance(expected, float):
        return abs(actual - expected) < FLOAT_THRESHOLD
    if isinstance(expected, str):
        ec = Counter(expected.strip().lower().split())
        ac = Counter(actual.strip().lower().split())
        et, at = ec.total(), ac.total()
        if et < 2 or at < 2:
            return expected == actual
        return (ec & ac).total() / (et + at) >= WORD_SIM_THRESHOLD
    return expected == actual


def parse_qwen3_coder_call(post: str, tool_schemas: dict) -> dict | None:
    """Parse the first qwen3-coder style tool call, with schema-based type coercion."""
    m = re.search(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", post, re.S)
    if not m:
        return None
    name = m.group(1).strip()
    args: dict = {}
    props = ((tool_schemas.get(name) or {}).get("properties")) or {}
    for pm in re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", m.group(2), re.S):
        key, raw = pm.group(1).strip(), pm.group(2)
        ptype = (props.get(key) or {}).get("type")
        if ptype == "string":
            args[key] = raw
        elif ptype in ("integer", "number", "boolean", "object", "array") or ptype is None:
            try:
                args[key] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                args[key] = raw
        else:
            args[key] = raw
    return {"name": name, "arguments": args}


def score(text: str, expected: dict, tool_format: str = "hermes", tool_schemas: dict | None = None) -> tuple[float, bool]:
    """Score one generation against expected_action; returns (reward, truncated).

    Mirrors the server semantics: the first parseable <tool_call> block is the
    action; otherwise any post-think text is a chat message. A generation with
    neither a closed think block nor a tool call is treated as truncated
    (no action found -> reward 0).
    """
    think_end = text.find("</think>")
    post = text[think_end + len("</think>") :] if think_end >= 0 else text
    call = None
    if tool_format == "qwen3_coder":
        call = parse_qwen3_coder_call(post, tool_schemas or {})
    else:
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", post, re.S)
        if m:
            try:
                call = json.loads(m.group(1))
            except json.JSONDecodeError:
                call = None  # malformed block degrades to text, like the hermes parser
    if call is not None and "name" in call:
        if expected["type"] != "function_call":
            return 0.0, False
        if call["name"] != expected["name"]:
            return 0.0, False
        try:
            expected_args = json.loads(expected["arguments"])
        except json.JSONDecodeError:
            return 0.0, False
        actual_args = call.get("arguments")
        if isinstance(actual_args, str):
            try:
                actual_args = json.loads(actual_args)
            except json.JSONDecodeError:
                return 0.0, False
        return (1.0 if compare_arguments(expected_args, actual_args) else 0.0), False
    if think_end < 0:
        return 0.0, True  # ran out of tokens mid-think: no action found
    if expected["type"] == "message":
        return 1.0, False  # any chat message matches, content is not compared
    return 0.0, False


def responses_input_to_chat(items: list[dict]) -> list[dict]:
    """Convert Responses-API input items to chat messages (same as drift probe)."""
    messages: list[dict] = []
    for it in items:
        typ = it.get("type")
        role = it.get("role")
        if typ == "reasoning":
            continue
        if typ == "function_call":
            # The nemotron template applies `|items` to arguments (requires a
            # dict); the qwen template accepts both string and dict. Use dict.
            arguments = it["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": it.get("call_id") or it.get("id") or "call_0",
                            "type": "function",
                            "function": {"name": it["name"], "arguments": arguments},
                        }
                    ],
                }
            )
        elif typ == "function_call_output":
            messages.append(
                {"role": "tool", "tool_call_id": it.get("call_id", ""), "content": it.get("output", "")}
            )
        elif role in ("system", "user", "assistant"):
            content = it.get("content")
            if isinstance(content, list):
                content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            messages.append({"role": role, "content": content or ""})
    return messages


def main() -> None:
    """Shard, resume, generate, score, flush."""
    args = parse_args()
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))
    if args.template_mode == "recipe":
        cfg = yaml.safe_load(open(args.recipe_yaml))
        template_str = cfg["policy"]["generation"]["vllm_cfg"]["http_server_serving_chat_kwargs"]["chat_template"]
        env = jinja2.Environment()
        env.filters.setdefault("tojson", lambda v: json.dumps(v, ensure_ascii=False))
        template = env.from_string(template_str)

        def render(messages: list[dict], tools_cc: list[dict]) -> str:
            return template.render(messages=messages, tools=tools_cc, add_generation_prompt=True)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        def render(messages: list[dict], tools_cc: list[dict]) -> str:
            return tok.apply_chat_template(
                messages,
                tools=tools_cc,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )

    my_rows: list[tuple[int, dict]] = []
    with open(args.in_jsonl) as f:
        for i, line in enumerate(f):
            if i % shard_n == shard_k:
                my_rows.append((i, json.loads(line)))

    done = 0
    try:
        with open(args.out) as f:
            done = sum(1 for _ in f)
    except FileNotFoundError:
        pass
    todo = my_rows[done:]
    print(f"shard {args.shard}: {len(my_rows)} rows, {done} already scored, {len(todo)} to go")
    if not todo:
        return

    from vllm import LLM, SamplingParams

    llm_kwargs = {}
    if args.mamba_f32_cache:
        llm_kwargs["mamba_ssm_cache_dtype"] = "float32"
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=not args.no_enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        **llm_kwargs,
    )
    sp_kwargs = {}
    if args.bad_words:
        sp_kwargs["bad_words"] = args.bad_words.split(",")
    sp = SamplingParams(
        n=args.samples, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens, **sp_kwargs
    )

    out = open(args.out, "a")
    for start in range(0, len(todo), args.chunk):
        batch = todo[start : start + args.chunk]
        entries = []  # (idx, row, prompt-or-None)
        for idx, row in batch:
            rcp = row["responses_create_params"]
            messages = responses_input_to_chat(rcp["input"])
            tools = [
                {"type": "function", "function": {k: t.get(k) for k in ("name", "description", "parameters")}}
                for t in rcp.get("tools", [])
            ]
            prompt = render(messages, tools)
            entries.append((idx, row, prompt if len(prompt) <= args.max_prompt_chars else None))
        gen_prompts = [p for _, _, p in entries if p is not None]
        outputs = iter(llm.generate(gen_prompts, sp) if gen_prompts else [])
        for idx, row, prompt in entries:
            expected = row["expected_action"]
            if prompt is None:
                out.write(
                    json.dumps({"i": idx, "expected_type": expected["type"], "skipped": "prompt_too_long"})
                    + "\n"
                )
                continue
            req = next(outputs)
            tool_schemas = {
                t.get("name"): t.get("parameters") or {}
                for t in row["responses_create_params"].get("tools", [])
            }
            rewards, n_trunc = [], 0
            for comp in req.outputs:
                r, trunc = score(comp.text, expected, args.tool_format, tool_schemas)
                rewards.append(r)
                n_trunc += int(trunc)
            mean = sum(rewards) / len(rewards)
            var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            out.write(
                json.dumps(
                    {
                        "i": idx,
                        "expected_type": expected["type"],
                        "rewards": rewards,
                        "reward_mean": mean,
                        "reward_var": var,
                        "n_truncated": n_trunc,
                    }
                )
                + "\n"
            )
        out.flush()
        print(f"scored {min(start + args.chunk, len(todo))}/{len(todo)}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
