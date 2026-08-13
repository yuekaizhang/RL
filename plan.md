# DSpark Qwen3-8B Speculative-Decoding RL Co-Training on the Automodel (DTensor-v2) Path

## Goal Description

Add DSpark draft-model co-training to NeMo-RL's automodel (DTensor-v2) backend so that DAPO RL training of Qwen3-8B simultaneously trains the `deepseek-ai/dspark_qwen3_8b_block7` block drafter and refits it into vLLM for speculative-decoding generation, mirroring the existing Megatron EAGLE3 co-training pattern. Target model: Qwen3-8B; draft model: `deepseek-ai/dspark_qwen3_8b_block7` (block_size=7, 5 draft layers, target_layer_ids=[1,9,17,25,33]); cluster: 1 node × 8 GPUs; generation length: 2048 tokens. Deliverables are the core code, a no-speculation baseline recipe YAML, and a DSpark co-training recipe YAML.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Config schema and setup guards. `DraftConfig` gains `algo` ("eagle3" default | "dspark") and a `dspark` sub-block (`num_anchors`, `ce_loss_alpha`, `l1_loss_alpha`, `confidence_loss_alpha`, `loss_decay_gamma`, `train_embed_and_head`); exemplar defaults live in `examples/configs/grpo_math_1B.yaml` (YAML single source of truth, no hidden code defaults); architecture fields (`block_size`, `target_layer_ids`, `mask_token_id`, markov/confidence fields) come only from the draft checkpoint's `config.json`; the backend guard in `nemo_rl/models/policy/lm_policy.py` allows draft training on Megatron OR on DTensor-v2 with `algo: dspark`.
  - Positive Tests (expected to PASS):
    - Loading the DSpark recipe through the config loader yields a valid `PolicyConfig` with the draft block populated.
    - Existing EAGLE3 Megatron recipes continue to parse and pass the guard unchanged (`algo` defaults to eagle3).
  - Negative Tests (expected to FAIL / be rejected):
    - `algo: dspark` with `dtensor_cfg._v2: false` (or Megatron backend) → hard setup error.
    - `algo: dspark` with any of: TP > 1, CP > 1, sequence packing enabled, activation checkpointing enabled, LoRA enabled, `model_name: null` → hard setup error.
    - Draft checkpoint `config.json` missing `block_size`, `target_layer_ids`, or `mask_token_id` → hard error.
    - `confidence_loss_alpha > 0` while the loaded draft config has the confidence head disabled → hard error.
- AC-2: Vendored DSpark package. `nemo_rl/models/automodel/draft/` contains the Qwen3 DSpark model, anchor/mask/noise utilities, markov head, sampling helpers (`_sampling.py` is required by the model), and the loss module, each file carrying a current-year NVIDIA copyright header plus a lineage header naming the source repo, path, and commit.
  - Positive Tests: `ruff check` passes on the package; importing the package succeeds without the Automodel submodule containing dspark; a lineage header is present in every vendored file.
  - Negative Tests: `grep` for imports from `nemo_automodel.components.speculative` in nemo_rl returns nothing (the vendored code must be self-contained); missing copyright header fails the repo's header check.
- AC-3: Worker integration and checkpointing. `DTensorPolicyWorkerV2` builds the draft from the pretrained checkpoint before optimizer construction; a thin composite `nn.Module` with named children {policy, draft} backs a single optimizer (stable [policy, draft] param-group order) and is the paired `model` argument for optimizer state save/load; policy weights save/load exactly as today; draft weights save/load as a separate draft entry with dspark meta (`mask_token_id`, `block_size`, `target_layer_ids`) and an optimizer-layout record (group names/order, per-group param counts); gradient clipping covers policy + draft modules globally while `draft_grad_norm` is reported separately; reference-policy weight swaps and eval paths never touch the draft.
  - Positive Tests: save → resume reproduces identical draft weights, optimizer state, and next-step draft loss; refit and inference paths reference the policy module unchanged.
  - Negative Tests: resume with a checkpoint whose dspark meta or optimizer-layout record mismatches the current config → hard error; a checkpoint saved without the draft entry cannot silently resume a dspark run.
- AC-4: Hidden capture and raw-logit tap. Forward hooks on the policy's decoder layers at `target_layer_ids` and on the final norm capture detached `target_hidden_states [B,S,5H]` and `target_last_hidden_states [B,S,H]` once per training forward (last-write-wins, idempotent); teacher logits for distillation are taken raw, before temperature scaling in the automodel forward path.
  - Positive Tests: captured tensor shapes match [B,S,len(target_layer_ids)·H] and [B,S,H]; teacher logits equal the model's pre-scaling logits.
  - Negative Tests: capture is inactive (no hooks fire, no memory held) when `draft.enabled: false` or during logprob/eval forwards.
- AC-5: DSpark loss in the RL step. A `DSparkLossWrapper` (parallel to `DraftLossWrapper`) and a dspark branch in the automodel `LossPostProcessor` compute `total = policy_loss + loss_weight × (ce_α·CE + l1_α·TV + conf_α·BCE)` with: anchors sampled from the response region (`min(num_anchors, valid response length)` per sequence); alignment where block offset k of anchor a predicts the token at position a+1+k supervised by teacher logits gathered at a+k; TV computed chunked against detached raw policy logits; confidence BCE against the detached per-position acceptance rate; all terms normalized by anchor-token counts all-reduced over the worker's DP process group (no internal world-size multiply — the outer `automodel_forward_backward` dp·cp scaling provides compensation); dummy microbatches contribute zero to numerators and denominators while still joining the all-reduce; metrics (`draft_loss`, `draft_ce_loss`, `draft_tv_loss`, `draft_conf_loss`, `draft_accept_rate@k`, `draft_tau`, `draft_grad_norm`) always present and shape-stable.
  - Positive Tests: with `loss_weight: 0` training metrics match the baseline run bit-for-bit in policy loss; draft gradients are nonzero when `loss_weight > 0`; hiddens/teacher detachment means policy-trunk gradients are identical with draft enabled vs disabled.
  - Negative Tests: a microbatch with zero valid response tokens produces zero draft loss and no NaN; enabling sequence packing with dspark is rejected before the loss path is reached.
- AC-6: Refit into vLLM. `prepare_refit_info` and the DTensor params generator emit all draft weights (including trained `embed_tokens`/`lm_head`, `fc`, `hidden_norm`, `norm`, layers, markov head, confidence head) as `draft.<hf_name>` across all refit transports; the vLLM extension locates the dspark drafter at `model_runner.speculator.model` behind a hard runtime assertion and loads the streamed weights; ranks that own a speculator hard-error on key mismatch while ranks without one skip; a warning fires when `speculative_config.num_speculative_tokens != draft block_size`.
  - Positive Tests: after the first refit the drafter's `has_own_embed_tokens`/`has_own_lm_head` are set and generation produces nonzero `vllm/spec_acceptance_length`; refit info and every transport stream identical `draft.*` key sets.
  - Negative Tests: a renamed/missing draft key on a speculator-owning rank → hard error, not a skip; a vLLM build without the expected speculator attribute → assertion naming the attribute searched.
- AC-7: Recipe YAMLs. `examples/configs/recipes/llm/dapo-qwen3-8b-1n8g-automodel.yaml` (baseline: Qwen/Qwen3-8B, MoE config removed, backend te/te/torch_fp32 with `enable_hf_state_dict_adapter: true`, 2048 input / 2048 generation / 4096 total, vLLM TP=1, `gpu_memory_utilization: 0.5`, `enforce_eager: true`, activation checkpointing off, DAPO block inherited) and `examples/configs/recipes/llm/dapo-qwen3-8b-1n8g-automodel-dspark.yaml` (inherits baseline; adds the `policy.draft` block and `speculative_config` with method dspark, `num_speculative_tokens: 7`, `attention_backend: FLASH_ATTN`, `draft_sample_method: probabilistic`).
  - Positive Tests: both YAMLs load through the config loader; the dspark YAML differs from baseline only in the draft block and generation speculative config.
  - Negative Tests: the baseline YAML contains no speculative or draft keys; neither YAML introduces MoE keys.
- AC-8: Smoke validation (runs submitted by the user via sbatch). Baseline trains several steps normally; the dspark run shows (a) `draft_loss` components decreasing, (b) `vllm/spec_acceptance_length` (τ) **≥ 2 at smoke — hard requirement per user decision**, (c) no acceptance cliff after the first refit, (d) a recorded throughput delta vs baseline.
  - Positive Tests: τ ≥ 2 within the smoke run; acceptance drifts smoothly across refits.
  - Negative Tests: τ < 2 or an acceptance cliff after refit means the implementation is not accepted until root-caused (first suspects: refit key mapping, embed/lm_head streaming, loss alignment).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation delivers the vendored DSpark package with lineage documentation, full DTensor-v2 worker integration (composite-module optimizer checkpointing, layout-meta validation, separate draft grad-norm reporting), raw-logit teacher tap, chunked TV loss, complete refit coverage across all three transports, vLLM dspark drafter loading with runtime assertions, both recipe YAMLs, exemplar config defaults, and all listed hard guards — verified by ruff, config parsing, and the two smoke runs.

### Lower Bound (Minimum Acceptable Scope)

The implementation trains the DSpark draft on rollout data inside the DAPO step with the full three-term loss, checkpoints and resumes it correctly, refits all draft weights into vLLM's dspark drafter on at least the IPC-ZMQ path with the other transports sharing the same generator, ships both YAMLs, and enforces every hard guard in AC-1. All eight acceptance criteria still hold; the lower bound trims only implementation polish (e.g., diagnostics beyond the required metrics), never guards or correctness.

### Allowed Choices

- Can use: forward hooks or an equivalent capture mechanism producing the same detached tensors; any internal structure for the composite module as long as optimizer pairing and checkpoint layout validation hold; chunk sizes / gather strategies for TV as long as memory stays bounded; porting capture logic from `../Automodel` target wrapper.
- Cannot use: bumping the Automodel submodule; importing dspark from `nemo_automodel` (must be vendored); patching vLLM source for the drafter (verified unnecessary); training the draft through a second optimizer; silent fallbacks on any misconfiguration; sequence packing, TP > 1, CP > 1, activation checkpointing, or LoRA on the dspark path in this iteration.
- Fixed per the design (no choice): full three-term dspark loss with pretraining hyperparameters as YAML defaults; draft embed/lm_head unfrozen and streamed on refit; teacher = detached raw policy logits; anchors from the response region; `draft.` weight-name prefix; the two recipe filenames.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

One workable path: vendor the dspark modules first and get them importable and ruff-clean. Then extend the worker setup — build the draft, wrap policy+draft in the composite, construct the optimizer over the composite, and only then run checkpoint resume. Register capture hooks around the training forward only. In the loss post-processor, gather teacher logits at anchor prediction positions before the policy loss consumes logits, run the draft forward with the flex-attention block mask, and compute the adapted loss. Extend the refit generator last, then the vLLM extension's drafter lookup. Recipes and static validation close it out.

### Relevant References

- `nemo_rl/models/megatron/draft/` — the EAGLE3 co-training pattern being mirrored (hidden capture, draft attach, export with `draft.` prefix).
- `nemo_rl/algorithms/loss/wrapper.py` — `DraftLossWrapper`, the structural template for `DSparkLossWrapper`.
- `nemo_rl/models/automodel/train.py` — `automodel_forward_backward` (dp·cp loss scaling before backward) and `LossPostProcessor` (where the dspark branch and raw-logit tap land).
- `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py` — setup, train loop, `scale_grads_and_clip_grad_norm` call, `prepare_refit_info`, params generator.
- `nemo_rl/models/automodel/checkpoint.py` — checkpoint manager whose model/optimizer pairing motivates the composite module.
- `nemo_rl/models/generation/vllm/vllm_backend.py` — `draft.` split, `_load_draft_weights`, `_get_drafter_model`.
- `nemo_rl/models/generation/__init__.py` — `configure_generation_config` / `has_refit_draft_weights` (dummy load_format + first-refit push, already wired via `run_grpo.py`).
- `../Automodel/nemo_automodel/components/speculative/dspark/` — vendoring source (model, common, markov_head, loss, `_sampling.py`, config builder) and `target.py` capture logic.
- `../Automodel/nemo_automodel/recipes/llm/train_dspark.py` — FSDP2 sharding pattern for the draft.
- Pinned vLLM 0.25.1: `vllm/model_executor/models/qwen3_dspark.py` (optional embed/lm_head loading), `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`, `vllm/v1/worker/gpu/model_runner.py` (`speculator` attribute).
- `examples/configs/recipes/llm/dapo-nanov3.5-30BA3B-4n8g-automodel.yaml` — baseline YAML template.
- `docs/superpowers/specs/2026-08-13-dspark-qwen3-8b-automodel-rl-design.md` — the approved design this plan implements.

## Dependencies and Sequence

### Milestones

1. Milestone 1 — Vendoring and config surface: vendored package (AC-2), config schema + guards + exemplar defaults (AC-1). No dependency on worker changes.
2. Milestone 2 — Trainer integration: hidden capture + raw-logit tap (AC-4), worker setup with composite module, FSDP2, clipping, checkpointing (AC-3). Depends on Milestone 1.
3. Milestone 3 — Loss: `DSparkLossWrapper` + `LossPostProcessor` branch with adapted normalization and alignment (AC-5). Depends on Milestone 2.
4. Milestone 4 — Refit and vLLM receive side (AC-6). Depends on Milestone 2 (worker), independent of Milestone 3.
5. Milestone 5 — Recipes and static validation (AC-7). YAMLs depend on Milestone 1 schema; static validation depends on all code milestones.
6. Milestone 6 — Smoke validation and analysis (AC-8). Depends on all previous milestones; runs are user-submitted.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Vendor dspark modules (model, common, markov_head, loss, `_sampling.py`, qwen3 config builder) into `nemo_rl/models/automodel/draft/` with lineage + copyright headers | AC-2 | coding | - |
| task2 | Extend `DraftConfig` (`algo`, `dspark` sub-block), exemplar YAML defaults, relax backend guard, add all dspark setup asserts (TP/CP/packing/AC/LoRA/model_name/confidence-consistency/config.json fields) | AC-1 | coding | task1 |
| task3 | Implement automodel hidden capture (layer + final-norm hooks, detached, training-forward only) and the raw pre-temperature teacher-logit tap | AC-4 | coding | task1 |
| task4 | Worker integration: draft build before optimizer, composite policy+draft module, single optimizer with stable param groups, FSDP2 sharding, global clipping incl. draft with separate `draft_grad_norm`, checkpoint save/resume with dspark meta + optimizer-layout record, reference/eval isolation | AC-3 | coding | task2 |
| task5 | `DSparkLossWrapper` + `LossPostProcessor` dspark branch: anchor sampling, alignment, chunked TV vs detached raw policy logits, confidence BCE, DP-group global denominators, dummy-batch zero contract, metrics | AC-5 | coding | task3, task4 |
| task6 | Refit: `prepare_refit_info` + params generator emit `draft.*`; vLLM `_get_drafter_model` dspark branch (`model_runner.speculator.model` assertion), owning-rank hard-error semantics, block_size consistency warning | AC-6 | coding | task4 |
| task7 | Write `dapo-qwen3-8b-1n8g-automodel.yaml` and `dapo-qwen3-8b-1n8g-automodel-dspark.yaml` | AC-7 | coding | task2 |
| task8 | Static validation: ruff check/format on all touched files; parse both YAMLs through the config loader | AC-1, AC-7 | coding | task5, task6, task7 |
| task9 | Independent full-diff review against this plan's acceptance criteria (loss scaling, alignment, checkpoint pairing, refit key coverage) | all | analyze | task8 |
| task10 | Analyze smoke-run results (baseline + dspark): draft_loss trajectory, τ ≥ 2 hard gate, refit-cliff check, throughput delta; root-cause guidance if gates fail | AC-8 | analyze | task9 |

## Claude-Codex Deliberation

### Agreements

- Vendoring (not submodule bump), DTensor-v2-only with hard guards (TP=1, CP=1, packing off, AC off, no LoRA, no draft on eval/reference/teacher paths).
- DSpark loss normalized by DP-all-reduced anchor-token counts with no internal world-size multiply, relying on the existing `automodel_forward_backward` dp·cp scaling.
- Draft built before optimizer construction/resume; global clipping over policy+draft with reporting-only separate draft grad norm.
- Teacher-logit alignment: block offset k of anchor a predicts token a+1+k, supervised by teacher logits from position a+k.
- Raw (pre-temperature) policy logits as the distillation teacher.
- Recipe naming matches existing sibling automodel recipes; unit/nightly tests deferred per explicit user scope decision, with ruff + config parse + smoke runs as minimum validation.
- vLLM receive side verified against the pinned wheel source; drafter probe kept as a hard runtime assertion.

### Resolved Disagreements

- Loss double-scaling (Codex round 1): Codex flagged that vendoring the Automodel loss unchanged double-scales gradients; Claude verified `automodel_forward_backward` multiplies by dp·cp before backward and adapted the vendored loss to the repo convention. Resolution: adapted normalization (accepted).
- Optimizer/checkpoint pairing (Codex rounds 1–2): "second param group naturally checkpoints" was rejected by Codex against the actual `AutomodelCheckpointManager` pairing semantics. Resolution: composite policy+draft module paired with the single optimizer, separate weight entries, layout-meta validation (accepted as prescribed).
- Temperature-scaled vs raw teacher logits (Codex round 1): Claude's v1 tapped logits inside `LossPostProcessor` (post-scaling); Codex required raw. Resolution: raw tap before `apply_temperature_scaling` (accepted; DAPO rollouts here use temperature 1.0 so the distinction is currently latent but principled).
- `num_speculative_tokens` consistency check: Claude's v1 compared against block_size−1; Codex corrected to block_size (block7 drafts 7 tokens). Resolution: compare against `block_size` (accepted).
- Recipe naming with parallelism markers (Codex round 1): rejected — existing automodel recipes carry no such markers; names match siblings.
- Unit/nightly tests in scope (Codex round 1): rejected for this iteration by explicit user decision; Codex agreed in round 2 given the fixed scope.
- Pretrained-only draft init: made explicit — `model_name` is mandatory for dspark; from-scratch draft init is out of scope.

### Convergence Status

- Final Status: `converged` (3 rounds; round 3 reported no disagreements, no required changes, nothing unresolved).

## Pending User Decisions

- DEC-1: Is the smoke-run τ (`vllm/spec_acceptance_length`) target of 2–4 a hard acceptance gate or an observational reference?
  - Claude Position: observational reference (recommended) — require nonzero and cliff-free, treat 2–4 as plausibility guidance.
  - Codex Position: N/A - open question (metric classification required by the planning workflow).
  - Tradeoff Summary: a hard gate makes AC-8 deterministic but couples acceptance to model/data behavior beyond code correctness; a soft reference risks shipping an under-performing integration unnoticed.
  - Decision Status: **User decided: hard requirement.** τ ≥ 2 at smoke is a hard acceptance gate (AC-8); τ < 2 means the implementation is not accepted until root-caused and fixed.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Repository Conventions
- New Python files need current-year NVIDIA copyright headers; vendored files additionally carry a lineage header (source repo, path, commit).
- Commit with `git commit -s --no-verify` on this cluster (pre-commit's uv sync has no disk space); run `uv run --no-sync ruff check` and `ruff format` manually before committing.
- YAML is the single source of truth for config defaults; never introduce `cfg.get(key, default)`-style hidden defaults for the new fields.

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# DSpark Qwen3-8B Speculative-Decoding RL Co-Training on the Automodel Path — Design

Date: 2026-08-13
Status: Approved (design review complete; pending spec review)

## Goal

Add DSpark draft-model co-training to NeMo-RL's automodel (DTensor-v2) backend, so
that DAPO RL training of Qwen3-8B simultaneously trains the
`deepseek-ai/dspark_qwen3_8b_block7` block drafter and refits it into vLLM for
speculative-decoding generation — mirroring the existing Megatron EAGLE3
co-training path.

Target: Qwen3-8B. Draft: `deepseek-ai/dspark_qwen3_8b_block7` (block_size=7,
5 draft layers, `target_layer_ids=[1,9,17,25,33]`). Cluster: 1 node × 8 GPUs.
Generation length: 2048 tokens.

## Background

- **DSpark** is a semi-autoregressive block drafter: given target hidden states
  (concat of `target_layer_ids` layers, projected by `fc` + `hidden_norm` as
  cross-attention K/V context), one parallel forward predicts a whole
  `block_size`-token block from an anchor token followed by `block_size-1` mask
  tokens. A Markov head adds intra-block token dependency; a confidence head
  predicts per-position acceptance. No TTT/recursive training — one forward with
  many randomly sampled anchor blocks via a flex-attention block mask.
- The repo already has EAGLE3 co-training on the **Megatron** backend
  (`nemo_rl/models/megatron/draft/`, `DraftLossWrapper`, `draft.`-prefixed refit
  stream, vLLM-side drafter loading in `vllm_backend.py`). The vLLM receiving
  side is backend-agnostic; the trainer-side draft path is gated to Megatron at
  `nemo_rl/models/policy/lm_policy.py:120`.
- DSpark training code exists in two sibling repos:
  `../DeepSpec` (research repo, offline hidden-state cache) and
  `../Automodel` (productionized `TrainDSparkRecipe` with online capture,
  FSDP2). The **pinned** Automodel submodule (`24b47e85`) does NOT contain the
  dspark components, and must stay pinned (the nano-v3.5 recipe depends on it),
  so dspark modules are **vendored** into nemo_rl rather than imported.
- The pinned vLLM (0.25.1) supports `speculative_config.method: "dspark"`
  (verified by the user via `vllm serve`).

## Approach (chosen: A)

Mirror the Megatron EAGLE3 integration pattern on the DTensor-v2 worker, with
dspark modules vendored from `../Automodel`.

Rejected alternatives:
- **B — bump the Automodel submodule** to a dspark-carrying commit: submodule
  bump changes MTP auto-build behavior (`1f13c9aa+`) and breaks the nano-v3.5
  recipe's premise; risk outweighs the benefit of avoiding ~2k vendored lines.
- **C — separate draft-trainer worker group**: diverges from the established
  co-training style, requires shipping hidden states across workers, doubles
  infra.

## §1 Config schema

Extend `DraftConfig` (`nemo_rl/models/policy/__init__.py:453`), keeping EAGLE3
compatibility:

```yaml
policy:
  draft:
    enabled: true
    model_name: deepseek-ai/dspark_qwen3_8b_block7
    loss_weight: 1.0
    algo: dspark            # new: "eagle3" (default) | "dspark"
    dspark:                 # new sub-block, read only when algo=dspark
      num_anchors: 512
      ce_loss_alpha: 0.1
      l1_loss_alpha: 0.9
      confidence_loss_alpha: 1.0
      loss_decay_gamma: 4.0
      train_embed_and_head: true
```

Architecture parameters (`block_size`, `target_layer_ids`, `mask_token_id`,
`markov_rank`, `markov_head_type`, confidence-head flags) are read from the
draft checkpoint's `config.json` — single source of truth, never duplicated in
YAML.

The guard at `lm_policy.py:120` is relaxed from "Megatron only" to
"Megatron, or DTensor-v2 (`dtensor_cfg._v2`) with `algo: dspark`".

## §2 Worker integration and hidden capture

New package `nemo_rl/models/automodel/draft/`, vendored from
`../Automodel/nemo_automodel/components/speculative/dspark/`:

- `dspark_model.py` — `Qwen3DSparkModel` (5 cross-attending draft layers, `fc`,
  `hidden_norm`, `markov_head`, `confidence_head`)
- `dspark_common.py` — anchor sampling, flex-attention block mask, noise
  embedding, eval mask
- `dspark_loss.py` — `compute_dspark_loss` (CE + TV + confidence BCE)
- `hidden_capture.py` — new, automodel-specific (below)

Worker changes (`dtensor_policy_worker_v2.py`, setup):

1. **Build**: `Qwen3DSparkModel.from_pretrained(draft.model_name)`, bf16.
   `embed_tokens` and `lm_head` are **unfrozen and trained** (user decision;
   `train_embed_and_head: true`). All draft params `requires_grad=True`.
2. **FSDP2**: `fully_shard` per draft decoder layer + root, same pattern as
   `../Automodel/train_dspark.py:1084`, reusing the policy's DP mesh.
3. **Optimizer**: draft params join the policy optimizer as a second param
   group (same LR schedule). DTensor has no Megatron `grad_norm_group`
   mechanism, so gradient clipping stays global; `draft_grad_norm` is computed
   and reported separately after `scale_grads_and_clip_grad_norm`. (Documented
   behavioral difference vs. the Megatron path, which clips draft separately.)
4. **Checkpointing**: draft state saved/restored with the worker checkpoint in
   a separate `draft/` subdirectory (DCP). Policy checkpoints exclude `draft.`
   keys; reference-policy weight swaps skip the draft (parity with
   `megatron_policy_worker.py:1708`).

Hidden capture:

- Forward hooks on `model.layers[i]` for `i in target_layer_ids` (HF offset
  semantics matching `../Automodel/target.py:150`; `-1` = embedding output) plus
  the final `model.norm` hook for the post-final-norm last hidden state.
- Captured tensors are **detached**: `target_hidden_states [B,S,5H]`,
  `target_last_hidden_states [B,S,H]`. Detachment means the draft loss does not
  backprop into the policy trunk — the policy is updated only by the RL loss
  (parity with the EAGLE3 path's detached teacher logits).
- Constraints, asserted loudly at setup when `algo=dspark`: TP == 1, CP == 1,
  sequence packing disabled. Dynamic batching (padded batches) is supported via
  the loss mask.

Train-step order (inside `train()`, one `automodel_forward_backward` call):
policy forward (hooks active) → RL loss → draft forward in the same graph
(cached hiddens + rollout input_ids, flex-attention anchor blocks) → dspark
loss → `total = policy_loss + λ·draft_loss` → single backward. Integration via
the loss wrapper (§3); `automodel_forward_backward`'s signature is unchanged.

The draft forward is independent of the policy's automodel backend config
(pure HF + flex_attention module); FSDP2 + flex_attention is already validated
in `../Automodel`.

## §3 Loss details and alignment

**Anchor sampling**: anchors are sampled from the **response region**
(`token_mask == 1`, i.e., RL-generated tokens); the prompt serves only as
attention context. Per-sequence anchor count =
`min(num_anchors, valid response length)`, reusing `sample_anchor_positions`.
Each anchor expands into one block (anchor token + `block_size-1` mask tokens)
under the flex-attention block mask (each block sees context strictly before
its anchor, plus itself).

**Loss terms** (vendored `compute_dspark_loss`; the only change is the TV
target source):

1. **CE (α=0.1)**: cross-entropy of draft logits vs. actual rollout next
   tokens, weighted by the contiguous-prefix `eval_mask` (cumprod) × positional
   decay `exp(-k/γ)`, γ=4.0. Rollout tokens are target-sampled by construction,
   matching the pretraining data premise.
2. **TV distillation (α=0.9)**: `½‖softmax(draft_logits) −
   softmax(teacher_logits)‖₁`, where **teacher_logits are the policy's own
   logits from the training forward (detached), gathered at each block's
   prediction positions**. This replaces pretraining's
   `draft_lm_head(target_last_hidden)`: with the draft lm_head unfrozen, a
   self-referenced target would not be self-consistent. `target_last_hidden`
   capture is retained for diagnostics only. Gathering happens on the already
   materialized `[B,S,V]` bf16 logits — no new memory peak.
3. **Confidence BCE (α=1.0)**: confidence-head output vs. the detached
   per-position acceptance rate `1 − ½‖p_draft − p_teacher‖₁` (pretraining
   formula unchanged).

**Normalization**: all three terms use a global denominator — valid
anchor-token counts all-reduced across DP ranks (the repo's
`global_valid_toks` convention, matching dspark's own global denominators at
`DeepSpec loss.py:227`) — so micro-batch splits do not change gradient scale.

**Wrapper**: new `DSparkLossWrapper` in `nemo_rl/algorithms/loss/wrapper.py`,
structurally parallel to `DraftLossWrapper`:
`total = policy_loss + loss_weight × (ce_α·CE + l1_α·TV + conf_α·BCE)`.
The automodel `LossPostProcessor` (`nemo_rl/models/automodel/train.py`) gains a
dspark branch: when the worker passes a draft model + captured hiddens, it runs
the draft forward + dspark loss after the policy loss and returns the combined
value. Gradients flow to draft params via the λ-scaled sum; the policy trunk is
unaffected (hiddens detached).

**Metrics**: `draft_loss` and components (`draft_ce_loss`, `draft_tv_loss`,
`draft_conf_loss`), `draft_accept_rate@k` (k=1..7), `draft_tau` (expected
accepted length, trainer-side estimate), `draft_grad_norm`. Generation side
reuses the existing `vllm/spec_acceptance_length` / `vllm/spec_acceptance_rate`
counter aggregation (`nemo_rl/models/generation/vllm/utils.py:440`), which is
method-agnostic. Trainer-side τ vs. vLLM-measured acceptance is the primary
observable of the experiment.

## §4 Refit / vLLM weight sync

Export side (DTensor-v2 worker) — one central change:

- `prepare_refit_info` (`dtensor_policy_worker_v2.py:1074`): append draft
  entries after the policy table, keyed `draft.<name>`. The draft is a native
  HF module — no `_maybe_adapt_tensor_to_hf` adaptation needed. Expected keys:
  `fc.weight`, `hidden_norm.weight`, `layers.{i}.*`, `markov_head.*`,
  `confidence_head.proj.*`, `embed_tokens.weight`, `lm_head.weight`, `norm.weight`.
- `dtensor_params_generator` (`:90`): yield draft params at the end
  (FSDP2 DTensor → `full_tensor()`, `draft.` prefix). All three refit paths
  (IPC-ZMQ, collective broadcast, checkpoint-engine) share this generator —
  one change covers all.
- Cost note: with embed/lm_head trained, two ~1.2 GB bf16 tensors ride every
  refit (~1–2 s extra); accepted as the direct cost of
  `train_embed_and_head: true`.

Receive side (vLLM): reuse the `draft.` split (`vllm_backend.py:377`) and
`_load_draft_weights` (`:430`) machinery, with two dspark adaptations:

1. `_get_drafter_model` (`:419`): add a dspark branch. **Verified against the
   pinned vLLM 0.25.1 source**: the dspark drafter lives at
   `model_runner.speculator.model` (`vllm/v1/worker/gpu/model_runner.py:189–311`,
   `DSparkSpeculator` in `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`,
   model class `Qwen3DSparkForCausalLM` in
   `vllm/model_executor/models/qwen3_dspark.py`) — note the attribute is
   `speculator`, not `drafter` as in the eagle3 layout.
2. Key mapping: `Qwen3DSparkForCausalLM.load_weights` (qwen3_dspark.py:149)
   consumes HF-style names directly, including `markov_head` (mapped to
   `markov_embed`/`markov_bias` internally via the loader) — a thin rename
   layer, if any, is confirmed small at implementation time.

**Embed/lm_head sharing risk — RESOLVED, no vLLM patch needed**:
`Qwen3DSparkForCausalLM` owns its own `ParallelLMHead` (qwen3_dspark.py:112)
and its `load_weights` treats `embed_tokens`/`lm_head` as *optional*: when
present in the incoming weights they are loaded as the drafter's own (setting
`has_own_embed_tokens` / `has_own_lm_head`); only when omitted are they shared
from the target (qwen3_dspark.py:151–184). Streaming our trained
`draft.embed_tokens.weight` / `draft.lm_head.weight` therefore works as-is.

Startup/config flow — essentially unchanged:

- `run_grpo.py:114` already sets `has_refit_draft_weights` from
  `policy.draft.enabled`; `configure_generation_config` keeps
  `load_format: dummy`, and the first refit pushes real policy + draft weights.
- `speculative_config` passes verbatim through `generation.vllm_kwargs` into
  `vllm.LLM(...)` (`vllm_worker.py:370→592`); the `num_speculative_tokens`
  lookahead clamp and spec-decode counter collection are already generic.
- Megatron's `_trim_vocab_padding` is unnecessary on the DTensor path (HF
  weights carry no vocab padding).

## §5 Recipe YAMLs, error handling, validation

### Baseline: `examples/configs/recipes/llm/dapo-qwen3-8b-1n8g-automodel.yaml`

Derived from `dapo-nanov3.5-30BA3B-4n8g-automodel.yaml` (same
`defaults: ../../grpo_math_1B.yaml`, hence `dtensor_cfg._v2: true`):

- `model_name: Qwen/Qwen3-8B`; cluster 1 node × 8 GPUs.
- All MoE config removed (`expert_parallel_size`, `moe_parallelizer`,
  `experts: gmm`, `dispatcher: deepep`); `automodel_kwargs.backend` slimmed to
  `attn: te, linear: te, rms_norm: torch_fp32`, keeping
  `enable_hf_state_dict_adapter: true` (refit depends on it).
- Lengths: `max_input_seq_length: 2048`, `generation.max_new_tokens: 2048`,
  `max_total_sequence_length: 4096`; DAPO `max_response_length` follows
  `${policy.generation.max_new_tokens}`; `overlong_buffer_length: 512` kept.
- Generation: `vllm_cfg.tensor_parallel_size: 1`,
  `gpu_memory_utilization: 0.5`, `enforce_eager: true`.
- DAPO algorithm block (dynamic sampling, TIS, clip-higher, reward scaling),
  optimizer/LR, dynamic batching on / packing off — inherited unchanged.

### DSpark: `examples/configs/recipes/llm/dapo-qwen3-8b-1n8g-automodel-dspark.yaml`

Inherits the baseline; adds only:

```yaml
policy:
  draft:
    enabled: true
    model_name: deepseek-ai/dspark_qwen3_8b_block7
    algo: dspark
    loss_weight: 1.0
    dspark: {num_anchors: 512, ce_loss_alpha: 0.1, l1_loss_alpha: 0.9,
             confidence_loss_alpha: 1.0, loss_decay_gamma: 4.0,
             train_embed_and_head: true}
  generation:
    vllm_kwargs:
      speculative_config:
        method: dspark
        model: ${policy.draft.model_name}
        num_speculative_tokens: 7
        attention_backend: FLASH_ATTN
        draft_sample_method: probabilistic
```

### Error handling (always loud, never silent degradation)

- `algo=dspark` asserts TP == 1, CP == 1, sequence packing disabled.
- Draft `config.json` missing dspark fields (`block_size`, `target_layer_ids`,
  `mask_token_id`) → hard error.
- vLLM-side refit key mismatch → hard error, not skip.
- `num_speculative_tokens` inconsistent with checkpoint `block_size - 1` →
  warning.

### Validation plan (in order, after implementation)

1. Static: `ruff check` / `ruff format`; both YAMLs parse through the config
   loader.
2. Baseline smoke run (sbatch, submitted by the user): DAPO Qwen3-8B trains
   several steps normally — establishes the control.
3. DSpark smoke run, observing: (a) `draft_loss` components decreasing,
   (b) `vllm/spec_acceptance_length` non-zero and plausible (block7 τ expected
   to start in the 2–4 range), (c) no acceptance cliff after the first refit
   (indirect evidence of refit-weight correctness), (d) throughput delta vs.
   baseline.
4. Refit correctness focus: acceptance should drift slowly across training
   steps, not jump; a jump points first at the §4 embed/lm_head-sharing risk.

## Deliverables

1. Core code: vendored dspark modules, DTensor-v2 worker draft support,
   `DSparkLossWrapper`, refit extensions, vLLM-side dspark drafter loading.
2. `dapo-qwen3-8b-1n8g-automodel.yaml` (baseline, no speculation).
3. `dapo-qwen3-8b-1n8g-automodel-dspark.yaml` (co-training).

Out of scope for this iteration: unit tests, sbatch launcher scripts, sequence
packing / TP / CP support for the dspark path, Megatron-backend dspark.

## Key user decisions recorded

- Full dspark loss (CE + TV + confidence), not distillation-only.
- Draft embed_tokens and lm_head are unfrozen and trained (and therefore
  streamed on every refit).
- Deliverables limited to core code + the two YAMLs.

--- Original Design Draft End ---
