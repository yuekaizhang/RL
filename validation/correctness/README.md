# Full-batch AutoModel lookahead 1 correctness validation

This extends the PR #4963 convergence experiment with controller ownership,
off-policy tensor and recovery checks. The long run remains separate from these
instrumented runs; instrumentation timings must not be used as throughput results.

## Executed checks

- 548 existing sampler, rollout-pump, checkpoint race/recovery and refit recovery
  tests passed (`targeted.xml`). The GPU rollout integration case in
  `test_rollout_pump.py` was excluded; real AutoModel runs below cover the backend.
- Four new tests exercise the eight-prompt LA1 window: capacities 8 and 15 fail
  validation, 16 succeeds; a complete future batch cannot bypass an unfinished
  current batch, further admission waits, and selected IDs are consumed once.
- 28 TQ lifecycle tests passed, including a real TQ regression for attaching a
  fresh client to live integer rows and registering an additional field.
- Actual AutoModel/vLLM training and recovery use Qwen2.5-1.5B, eight prompts,
  four responses, full batch 32, one training GPU and one generation GPU,
  `in_order`, LA1, 16 buffered/in-flight groups, TIS `[0, 2]`, KV recomputation,
  seed 42, and the requested `nemo_rl.1030.sqsh` image on H100 GPUs.

The machine-readable results and selected evidence are in `results.json`.
Local raw evidence is under `real/`: controller events, published and selected
TQ rows, actual worker loss inputs, resident vLLM weight slices, synchronized
generation GPU execution intervals and JSON reports. Logs and JUnit reports are
in this directory. Large temporary correctness checkpoints are removed after
verification; the convergence run's resume checkpoint is preserved.

## What the real-run assertions cover

1. Dispatch can target `trainer_version + 1`; buffer occupancy never exceeds
   16. Selection matches `target_step` in order. Prompt/group admission and
   successful training consumption have no skips or duplicates within the
   completed training steps. Unconsumed lookahead work at the bounded run's end
   is reported as outstanding work, rather than counted as missing training.
2. Token IDs, masks and rollout logprobs match exactly between canonical
   publication, controller selection and actual worker loss inputs. Recomputed
   policy logprobs also match those inputs. The worker checks the actual tensor
   handed to the actor objective against masked token-level TIS on CUDA exactly;
   independent CPU reconstruction permits float32 exponential rounding.
3. The effective inference weights are checked against the saved training
   weights after refit. This compares 58 shared-layout parameter slices (14,848
   values), after the explicit FP32-to-BF16 inference precision conversion. It
   does not fingerprint every parameter or test EP16 expert remapping.
4. An exclusive checkpoint cut captured eight completed, unconsumed groups and
   eight groups with 32 already-issued generation RPCs awaiting output. Recovery
   restores the canonical rows, redispatches unfinished groups, and consumes the
   stable logical IDs exactly once. Canonical rows are reused in the real run;
   completed sibling reuse is also covered by the recovery matrix tests.
   Regenerating unfinished work is allowed. Work performed after a saved cut in
   the old process is not durable and can be redone.
5. RPC timelines show future generation overlapping policy forward/backward.
   The GPU probe additionally records CUDA event durations inside synchronized
   generation execute calls with nonzero scheduled tokens and requires complete
   such intervals inside policy forward/backward. This probe adds synchronization
   overhead and is only installed by this audit launcher.

The original EP16 Qwen3.5-35B-A3B run completed 600 steps and was compared with
the supplied synchronous and SC LA0 baselines through 500 steps. That comparison
remains a historical, single-seed comparison, not a same-image randomized A/B.
Online validation remains disabled at the user's instruction. The separate
600-to-900 continuation retains its original configuration.

## Recovery issue fixed

The fresh/restoring TQ client previously warmed every configured field with
float32 placeholder data. A restored controller can already hold live int64
rows; that placeholder write then logs a dtype mismatch. `register_partition`
now asks the pinned TQ client's metadata API which fields already exist and
warms only new fields. The regression test checks both integer-row preservation
and successful registration of a new column. A real checkpoint resume is also
checked for absence of the dtype error.

## Reproduce

Inside an allocated container with two available GPUs, set a fresh output
directory and execute:

```bash
SC_AUDIT_DIR=/lustre/path/to/fresh-audit bash validation/correctness/run.sh
```

The script stops after checkpoint 1, restarts the same target-six-step run, then
asserts the event/tensor/GPU checks and resident weight comparison. The actual
recorded run reached 10 steps and also resumed at steps 3, 6, 7 and 8 while developing the audit and
verifying the schema fix. One audit assertion initially omitted masked-token
sanitization; its uncommitted microbatch captures are retained separately in
`real/aborted-loss/` with an exclusion manifest. Only the durable prefix and
resumed successful updates contribute to the final exactly-once checks.

The audit classes, custom worker hook and vLLM extension are imported only by
`run_audited.py`. Ordinary training launchers do not install them.
