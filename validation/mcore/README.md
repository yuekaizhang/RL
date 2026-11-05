# MCore SC lookahead comparison for gen_kl_error

Run Qwen3.5-35B-A3B-Base with MCore/Megatron-Bridge policy training and the same
vLLM generation setup as the existing AutoModel experiment. Both cases target
300 updates. The paired configs differ only in lookahead and output/run identity.

| Setting | Both cases |
| --- | --- |
| Resources | 3 nodes × 8 GPUs; 16 training + 8 generation |
| Training parallelism | EP16, TP1, CP1, PP1 |
| Generation | vLLM TP8, BF16, same requested `nemo_rl.1030.sqsh` |
| Batch | 8 prompts × 4 responses = 32; microbatch 1 |
| Length | Input 2048, generated 2048, total 4096 |
| Data/seed | Same DAPOMath17K configuration; seed 42 |
| Optimizer | Adam, LR 1e-6, warmup 10 then constant, beta .9/.999, eps 1e-8, WD .1, clip 1 |
| Optimizer state | FP32 master/gradients, BF16 first/second moments, parameter remainders |
| Gradients | FP32 reduction, matching AutoModel's FSDP mixed precision policy |
| Loss | Existing GRPO settings, TIS [0,2], no reference KL penalty |
| Sampler | `in_order`; LA0 or LA1; 16 buffers/in-flight groups, full-batch threshold 8 |
| Router | Trainable, no auxiliary loss or bias updates; normalized top-k weights |
| Validation | Disabled, as in the existing experiment |
| Checkpoint | Every 10 updates; keep latest; stop/save at 3h45m of a 4h allocation |

Effective settings are exported to `config-equivalence.json`. The real provider
probe records the pinned Bridge and MCore source paths in `provider-probe.json`:
40 layers, hidden size 2048, 256 experts, top-k 8, EP16/TP1/CP1, frozen vision
encoder/projector. No model/provider implementation was changed for these runs.

The MCore normalized top-k option has different semantics from AutoModel's:
MCore `moe_router_pre_softmax=false` applies softmax to the top-k logits;
AutoModel applies full softmax, selects top-k and renormalizes. Setting the
MCore option true would omit renormalization and change the model computation.
The inherited HF override already disables the AutoModel router auxiliary loss
and MTP, so MCore disables both too. MCore's logprob chunking requires
`defer_fp32_logits=true`.

This aligns the main learning settings and holds all settings fixed between the
two MCore cases. Backend implementations still differ: MCore's standard optimizer
parameter groups exempt biases/vector parameters from weight decay, whereas the
AutoModel optimizer receives all trainable parameters in one group. Optimizer
sharding, master-weight storage and kernels also differ. Cross-backend curves are
diagnostic comparisons; they are not bitwise training parity or same-batch
counterfactual tests. The two MCore runs also generate their own sampled tokens.

## Execution and evidence

The allocated debug job used the requested image and passed a dense Qwen2.5-1.5B
MCore/vLLM smoke with full batches of 32. LA0 ran three steps at lag 0; LA1 ran
three steps at lags 0/1/1. A real LA1 resume from checkpoint 2 completed step 3,
restoring eight canonical groups and 16 unfinished groups with the correct LR.
All updates had 32 valid samples, finite metrics, and no dropped or masked groups.
Selected results are in `smoke-proof.json` (raw local logs under `results/`).

Smoke target steps must stay fixed across restart: MCore validates the optimizer
scheduler's weight-decay horizon. Increasing the smoke target from 3 to 5 was
rejected before training; the passing recovery used the original target 3. The
300-step jobs always use target 300 across all allocations.

`submit.sh` checks the smoke, provider and config evidence before submitting four
chained allocations per case. `run.py` resumes the same W&B run and checkpoint;
later allocations skip container startup once step 300 is complete. The first
full-model allocation starts with target 300; its initial updates are monitored
without stopping and resubmitting a separate long experiment.

Submitted on 2026-11-04:

- LA0: 21318164 → 21318165 → 21318166 → 21318167.
- LA1: 21318168 → 21318169 → 21318170 → 21318171.

W&B run IDs in `yuekaiz/nemo-rl-automodel-sc` are `yuekaiz4963mc0s42` and
`yuekaiz4963mc1s42`; they appear when their allocations start. All job and run names
have the requested `yuekaiz` prefix. The separate AutoModel 900-step chain remains
held at the user's explicit request and must not be released automatically.

`monitor.py` reads native metrics, checks ordering/lag/sample counts/LR/nonfinite
values, and writes `results/live-status.json`. Finite KL increases raise alerts;
they do not cancel the experiment. Structural or nonfinite failures cancel this
MCore pair. Figure generation executes inside a ready Slurm container; each
allocation also plots before exiting. `results/comparison.png` contains KL,
reward, response length, loss, entropy, gradient norm, validation/acc (unrecorded)
and step time, with historical AutoModel LA0/LA1 references through step 300.

Interpret the curves by comparing the MCore LA1–LA0 gap first. Similar late KL
growth on both backends supports an effect of stale-policy sampling; growth
isolated to one backend warrants a fixed-token backend/refit investigation.
Neither outcome alone establishes a bug. Check response length, reward, loss,
entropy, sampling corrections and throughput alongside KL.
