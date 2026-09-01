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
"""Compare train-vs-generation logprob mismatch across logged train_data JSONLs.

Each row of a NeMo-RL train_data_step*.jsonl carries per-token
``generation_logprobs`` (vLLM, at sampling time) and ``prev_logprobs`` (the
training backend's recompute). This script reports, per file:

- token_mult_prob_error = exp(masked mean |lp_train - lp_gen|)  (the wandb metric)
- gen_kl_error proxy    = masked mean (lp_train - lp_gen)
- per-sequence exp(mean|diff|) percentiles, to see the tail

Usage:
    python yuekai_scripts/compare_logprob_mismatch.py results/probe-*/logs/exp_001/train_data_step1.jsonl
"""

import json
import math
import sys


def analyze(path: str) -> None:
    tok_absdiff_sum = 0.0
    tok_diff_sum = 0.0
    tok_exp_sum = 0.0
    tok_count = 0
    seq_errors = []
    seq_moe_errors = []  # mean-of-exp: the actual token/seq_mult_prob_error metric
    outliers = []  # (|diff|, seq_idx, pos_frac, token_id)
    with open(path) as f:
        for seq_idx, line in enumerate(f):
            row = json.loads(line)
            gen = row["generation_logprobs"]
            prev = row["prev_logprobs"]
            mask = row["token_loss_mask"]
            # Some writers nest the per-token arrays one level deep.
            if gen and isinstance(gen[0], list):
                gen = [x for sub in gen for x in sub]
            if prev and isinstance(prev[0], list):
                prev = [x for sub in prev for x in sub]
            if mask and isinstance(mask[0], list):
                mask = [x for sub in mask for x in sub]
            tok_ids = row.get("token_ids") or []
            if tok_ids and isinstance(tok_ids[0], list):
                tok_ids = [x for sub in tok_ids for x in sub]
            n = min(len(gen), len(prev), len(mask))
            s_abs = s_diff = s_exp = 0.0
            s_n = 0
            seq_max = (0.0, 0, 0)  # (|diff|, pos, token_id)
            for i in range(n):
                if not mask[i]:
                    continue
                d = prev[i] - gen[i]
                a = abs(d)
                s_abs += a
                s_diff += d
                s_exp += math.exp(min(a, 60.0))
                s_n += 1
                if a > seq_max[0]:
                    seq_max = (a, i, tok_ids[i] if i < len(tok_ids) else -1)
            if s_n == 0:
                continue
            tok_absdiff_sum += s_abs
            tok_diff_sum += s_diff
            tok_exp_sum += s_exp
            tok_count += s_n
            seq_errors.append(math.exp(s_abs / s_n))
            seq_moe_errors.append(s_exp / s_n)
            outliers.append((seq_max[0], seq_idx, seq_max[1] / max(n, 1), seq_max[2]))

    seq_errors.sort()

    def pct(p: float) -> float:
        return seq_errors[min(len(seq_errors) - 1, int(p * len(seq_errors)))]

    print(f"{path}")
    print(f"  sequences: {len(seq_errors)}, masked tokens: {tok_count}")
    print(
        f"  token_mult_prob_error exp(mean|diff|): {math.exp(tok_absdiff_sum / tok_count):.4f}"
    )
    print(f"  mean signed diff (train - gen): {tok_diff_sum / tok_count:+.5f}")
    print(
        "  per-seq exp(mean|diff|) p50/p90/p99/max: "
        f"{pct(0.5):.3f} / {pct(0.9):.3f} / {pct(0.99):.3f} / {seq_errors[-1]:.3f}"
    )
    # NeMo-RL's token/seq_mult_prob_error metric is mean-of-exp (outlier
    # dominated), which is what the wandb curves and the seq mask filter use.
    seq_moe_errors.sort()

    def mpct(p: float) -> float:
        return seq_moe_errors[min(len(seq_moe_errors) - 1, int(p * len(seq_moe_errors)))]

    print(f"  token_mult_prob_error mean-of-exp (wandb metric): {tok_exp_sum / tok_count:.4f}")
    print(
        "  per-seq mean-of-exp p50/p90/p99/max: "
        f"{mpct(0.5):.3f} / {mpct(0.9):.3f} / {mpct(0.99):.3f} / {seq_moe_errors[-1]:.3f}"
    )
    outliers.sort(reverse=True)
    print("  worst per-seq token |diff| (top 5): ", end="")
    print(
        ", ".join(
            f"{a:.1f}@seq{s}/pos{pf:.2f}/tok{tid}" for a, s, pf, tid in outliers[:5]
        )
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
