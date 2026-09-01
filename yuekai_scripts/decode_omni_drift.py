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
"""Offline drift probe: decode pivot val prompts with a checkpoint and dump stats.

Renders prompts exactly like training serving does (the inlined omni chat
template from the recipe yaml, tools included, generation opened with <think>),
samples at the training temperature, then classifies each generation:
think length, action type (tool_call / message / malformed), and agreement with
expected_action. Run once per model and diff the summaries.

Example (inside a GPU container, vLLM worker venv python):
  python yuekai_scripts/decode_omni_drift.py \
    --model /path/to/hf_ckpt --val-jsonl datasets/conversational_tool_use_pivot/val.jsonl \
    --recipe-yaml examples/nemo_gym/grpo_qwen3_omni_30ba3b_thinking_tau1.yaml \
    --out results/omni_drift/step113.jsonl --n-prompts 64 --samples 4
"""

import argparse
import json
import re

import jinja2
import yaml


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--recipe-yaml", required=True, help="Recipe yaml holding the inlined chat template")
    p.add_argument("--out", required=True)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--tp", type=int, default=4)
    return p.parse_args()


def responses_input_to_chat(items: list[dict]) -> list[dict]:
    """Convert Responses-API input items to chat-completions messages.

    Mirrors the Gym model server translation closely enough for rendering:
    reasoning items are dropped (history thinking is not rendered for
    non-final turns by the template anyway), function calls become assistant
    tool_calls, function outputs become tool messages.
    """
    messages: list[dict] = []
    for it in items:
        typ = it.get("type")
        role = it.get("role")
        if typ == "reasoning":
            continue
        if typ == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": it.get("call_id") or it.get("id") or "call_0",
                            "type": "function",
                            "function": {"name": it["name"], "arguments": it["arguments"]},
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
                content = "".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            messages.append({"role": role, "content": content or ""})
    return messages


def classify(text: str, expected: dict) -> dict:
    """Classify one generation against the expected action."""
    think_end = text.find("</think>")
    think_len = think_end if think_end >= 0 else (len(text) if "<tool_call>" not in text else 0)
    post = text[think_end + len("</think>") :] if think_end >= 0 else text
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", post, re.S)
    action_type, name_match, args_exact, tool_name = "message", False, False, None
    if m:
        try:
            call = json.loads(m.group(1))
            tool_name = call.get("name")
            action_type = "tool_call"
            if expected.get("type") == "function_call":
                name_match = tool_name == expected.get("name")
                try:
                    args_exact = json.loads(expected.get("arguments", "{}")) == call.get("arguments")
                except (json.JSONDecodeError, TypeError):
                    args_exact = False
        except json.JSONDecodeError:
            action_type = "malformed_tool_call"
    elif not post.strip():
        action_type = "empty"
    type_match = (action_type == "tool_call") == (expected.get("type") == "function_call")
    return {
        "gen_chars": len(text),
        "think_chars": max(think_len, 0),
        "has_think_close": think_end >= 0,
        "action_type": action_type,
        "tool_name": tool_name,
        "type_match": type_match,
        "name_match": name_match,
        "args_exact": args_exact,
    }


def main() -> None:
    """Render, decode, classify, and dump."""
    args = parse_args()
    cfg = yaml.safe_load(open(args.recipe_yaml))
    template_str = cfg["policy"]["generation"]["vllm_cfg"]["http_server_serving_chat_kwargs"]["chat_template"]
    env = jinja2.Environment()
    env.filters.setdefault("tojson", lambda v: json.dumps(v, ensure_ascii=False))
    template = env.from_string(template_str)

    rows = []
    with open(args.val_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.n_prompts:
                break

    prompts, metas = [], []
    for row in rows:
        rcp = row["responses_create_params"]
        messages = responses_input_to_chat(rcp["input"])
        tools = [
            {"type": "function", "function": {k: t.get(k) for k in ("name", "description", "parameters")}}
            for t in rcp.get("tools", [])
        ]
        prompt = template.render(messages=messages, tools=tools, add_generation_prompt=True)
        prompts.append(prompt)
        metas.append(row["expected_action"])

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        max_model_len=16384,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        n=args.samples, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens
    )
    outputs = llm.generate(prompts, sp)

    agg: dict = {}
    with open(args.out, "w") as out:
        for req, expected in zip(outputs, metas):
            for comp in req.outputs:
                rec = classify(comp.text, expected)
                rec["expected_type"] = expected.get("type")
                rec["gen_tokens"] = len(comp.token_ids)
                rec["text"] = comp.text[:2000]
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                for k in ("gen_tokens", "think_chars"):
                    agg.setdefault(k, []).append(rec[k])
                for k in ("action_type",):
                    agg.setdefault(k, []).append(rec[k])
                for k in ("type_match", "name_match", "args_exact", "has_think_close"):
                    agg.setdefault(k, []).append(bool(rec[k]))

    n = len(agg["gen_tokens"])
    print(f"== {args.model} : {n} samples")
    print(f"mean gen tokens: {sum(agg['gen_tokens']) / n:.0f}")
    print(f"mean think chars: {sum(agg['think_chars']) / n:.0f}")
    from collections import Counter

    print("action types:", dict(Counter(agg["action_type"])))
    for k in ("type_match", "name_match", "args_exact", "has_think_close"):
        print(f"{k}: {100 * sum(agg[k]) / n:.1f}%")


if __name__ == "__main__":
    main()
