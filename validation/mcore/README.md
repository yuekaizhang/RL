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

## OOM recovery and active monitoring

Both original jobs reached step 11 and failed in step 12 while allocating a
4.26 GiB one-hot workspace in chunked logprob backward. Checkpoint 10 is the
durable restart point. Both cases now use `policy.logprob_chunk_size=512`
instead of 4096 and `expandable_segments:True`; batch size, sequence length,
parallelism, optimizer and target 300 are unchanged. Preserve the original
logs and annotate the execution change when comparing step time.

The GPU test `check_logprob_chunks.py` compares forward and backward results
using both chunk sizes. At sequence length 4096 and vocabulary 248320, maximum
logprob/gradient differences were 9.54e-7/1.19e-7; peak allocated memory for
this isolated computation decreased from 26.52 to 6.63 GiB. This is a workspace
measurement, not the complete model's memory footprint. See the local report
`results/logprob-chunk-check.json`.

MCore embeds optimizer state under its distributed weights checkpoint. The
resume launcher uses `CheckpointManager.get_resume_paths` to detect it rather
than requiring AutoModel's separate optimizer directory. Recovery jobs are
recorded in `slurm/submitted-chunk512.tsv` and appended to `submitted.tsv` so
plots retain steps from both allocations, using the latest record on retries.

The user requested checks every ten minutes. A user crontab entry invokes
`watch_10min.sh`, which queues a check in the original Codex conversation and
resumes it via the local daemon. The conversation target is stored only in
local `results/watch-target.json`. The minute-level metric monitor also invokes
this notifier when allocations end prematurely, metrics are invalid, or both
runs complete. Cron service and queue delivery were tested; local evidence is
in `results/cron-heartbeat.txt` and `results/watch-10min*.log`. The conversation
checks Slurm and actual logs, handles failures, and removes this schedule when
both cases finish. The held AutoModel 900-step jobs must remain held.

The first chunk512 retry also failed before any new update: loading compressed
Adam's portable FP32 state exhausted GPU memory in `FusedAdam.load_state_dict`.
Both cases therefore enable Bridge's existing
`checkpoint.stage_precision_aware_optimizer_state_on_cpu` through a new
NeMo-RL configuration pass-through. This changes checkpoint staging devices,
not the optimizer's training precision or saved values.
`check_optimizer_staging.py` verified bitwise-identical portable state and
next optimizer updates using the image's real FusedAdam; extra peak memory
fell from 224 to 96 MiB in the isolated test. This is not proof that the full
EP16 resume fits; inspect the new jobs before claiming recovery.
Latest retry IDs are in `slurm/submitted-cpu-staging.tsv`; all retries remain
in `submitted.tsv`. The submission helper checks that no prior MCore job
remains queued before creating another chain.

## Continuation from 300 to 600 (user requested 2026-11-05)

`la0_to600.yaml` and `la1_to600.yaml` extend the existing W&B runs to 600 updates.
The model, batch, lengths, topology, logprob chunk 512, optimizer and LR are
unchanged. They explicitly restore the saved scheduler using
`use_checkpoint_opt_param_scheduler=true`, `override_opt_param_scheduler=false`.
Its old horizon remains 300 updates; constant LR 1e-6 and constant WD 0.1
continue unchanged beyond that horizon. Warmup and optimizer counters resume.

`run_to600.py` initially reads the complete preserved checkpoint300, including
optimizer, scheduler, dataloader and native replay state. New checkpoints/logs
go to `results/yuekaiz-mcore-la{0,1}-600/`, so retention does not delete the
original 300-step checkpoint. Later allocations discover the latest new
checkpoint. An allocated preflight exercises the image's actual scheduler API
over synthetic updates301–600. Full optimizer restore and LR must be verified
at the first native step301; that training check is still pending.

Submitted chains (four 4-hour allocations each, with afterok dependencies):

- LA0: 21388471 → 21388472 → 21388473 → 21388474.
- LA1: 21388475 → 21388476 → 21388478 → 21388479.

The first chains failed before checkpoint loading because Megatron-LM's
`examples` package shadowed the repository entrypoint. The wrapper now loads
the exact source file. Recovery preflight21388461 passed both entries and
checkpoint discovery; full model/optimizer resume is still pending. Old IDs
are archived in `slurm/submitted-to600-attempt1.tsv`.

`slurm/submitted-to600.tsv` contains the current chains; all IDs also appear
in `submitted.tsv` for continuous plots. `submit_to600.sh --check` verifies
source checkpoint metadata/shards and recipe differences without submitting.
Completed cases skip startup in unused allocations; the monitor cancels unused
queued segments after a durable600 checkpoint and complete metric history.

Run `python -u -m validation.mcore.monitor --target-steps 600` on the login node.
It writes `results/live-status-to600.json`, checks metrics/queue each minute,
and uses `watch_to600.sh` to wake the original conversation every ten minutes
and on failures/alerts. No new crontab is used. The old300
`results/stop-10min-watch` remains in place; `results/stop-to600-watch` belongs
to this continuation and is written only after both600 targets finish.
Plotting runs inside allocations with `plot.py --target-steps 600`, producing
`results/comparison-to600.png`. Original300 report/plot remain available.
AutoModel fixed-token KL diagnosis continues independently; the AutoModel900
hold must not be released.
