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
    tok_count = 0
    seq_errors = []
    with open(path) as f:
        for line in f:
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
            n = min(len(gen), len(prev), len(mask))
            s_abs = s_diff = 0.0
            s_n = 0
            for i in range(n):
                if not mask[i]:
                    continue
                d = prev[i] - gen[i]
                s_abs += abs(d)
                s_diff += d
                s_n += 1
            if s_n == 0:
                continue
            tok_absdiff_sum += s_abs
            tok_diff_sum += s_diff
            tok_count += s_n
            seq_errors.append(math.exp(s_abs / s_n))

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


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
