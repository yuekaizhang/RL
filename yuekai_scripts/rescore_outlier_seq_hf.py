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
"""Tiebreaker: rescore outlier sequences with a plain HF transformers forward.

For sequences where vLLM (generation_logprobs) and the training backend
(prev_logprobs) disagree wildly at specific tokens, run the HuggingFace eager
implementation of the model over the exact same token ids and report its
logprob at those positions. Whichever engine HF agrees with is presumed
numerically sound; the other is the deviant.

Usage:
    python yuekai_scripts/rescore_outlier_seq_hf.py \
        --model /path/to/hf_ckpt \
        --jsonl results/probe-step600-mcore/logs/exp_001/train_data_step1.jsonl \
        --top 5
"""

import argparse
import json

import torch


def flat(x):
    return [v for s in x for v in s] if x and isinstance(x[0], list) else x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--top", type=int, default=5, help="How many outlier sequences to rescore.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.jsonl)]
    outliers = []  # (|diff|, seq_idx, pos, gen_lp, prev_lp)
    for seq_idx, row in enumerate(rows):
        gen = flat(row["generation_logprobs"])
        prev = flat(row["prev_logprobs"])
        mask = flat(row["token_loss_mask"])
        best = None
        for i in range(min(len(gen), len(prev), len(mask))):
            if not mask[i]:
                continue
            d = abs(prev[i] - gen[i])
            if best is None or d > best[0]:
                best = (d, seq_idx, i, gen[i], prev[i])
        if best is not None:
            outliers.append(best)
    outliers.sort(reverse=True)
    picks = outliers[: args.top]
    print("Rescoring outlier positions:", [(s, p) for _, s, p, _, _ in picks])

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    # The omni wrapper's forward unconditionally runs the vision tower; for
    # text-only scoring use the inner causal LM directly.
    lm = getattr(model, "language_model", model)

    for d, seq_idx, pos, gen_lp, prev_lp in picks:
        row = rows[seq_idx]
        token_ids = flat(row["token_ids"])
        # Truncate a bit after the outlier position: logprob at pos only
        # depends on tokens < pos, so this is exact and saves memory.
        ids = torch.tensor(token_ids[: pos + 1], dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.no_grad():
            logits = lm(input_ids=ids).logits  # [1, T, V]
        # prev/generation logprobs arrays are aligned so that index i holds the
        # logprob OF token i given tokens < i; HF logits at i predict token i+1.
        target = token_ids[pos]
        lp = torch.log_softmax(logits[0, pos - 1].float(), dim=-1)[target].item()
        tok_str = tok.decode([target])
        print(
            f"seq{seq_idx} pos{pos} token={target!r}({tok_str!r}): "
            f"vllm={gen_lp:.3f} train_backend={prev_lp:.3f} hf_{args.dtype}={lp:.3f}"
        )


if __name__ == "__main__":
    main()
