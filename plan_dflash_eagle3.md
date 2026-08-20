# DFlash + EAGLE3 Qwen3-8B Speculative-Decoding RL Co-Training (Automodel Path)

## Goal Description

Extend the DTensor-v2 (automodel) draft co-training path of NeMo-RL — today
dspark-only — to two additional drafter families, each with a runnable 1n8g
GRPO recipe against target `Qwen/Qwen3-8B`:

- **DFlash** (`RedHatAI/Qwen3-8B-speculator.dflash`): reuse the existing
  DSparkRuntime — the vendored `Qwen3DSparkModel` with `markov_rank=0` and no
  confidence head is exactly DFlash — plus speculators-format adapter
  extensions (accept `DFlashDraftModel`, map the anchor-inclusive
  `block_size 8` to the vendored predicted-positions convention of 7).
- **EAGLE3** (`RedHatAI/Qwen3-8B-speculator.eagle3`): new `Eagle3Runtime`
  built on a vendored copy of the speculators eagle3 model with TTT-unrolled
  training (ttt_steps=3, per-step soft-CE, `prev_correct` masking), teacher
  targets taken from the policy's stashed raw logits (d2t-offset-mapped)
  instead of frozen verifier-head copies.

Both recipes inherit `examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-automodel.yaml`
(non-thinking, 2048-token generation) with knobs aligned to the dspark recipe
for cross-method comparison. `train_embed_and_head: true` for both. vLLM refit
generalizes from dspark-only to `{dspark, dflash, eagle3}`, including
disabling vLLM's draft/target embed-and-head module sharing on the eagle path
(same hazard already fixed for dspark). Scope is experimental (qwen3-family
targets; RedHat speculators-format and deepseek flat-format checkpoints); the
megatron-backend eagle3 path stays untouched.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Speculators adapter supports dflash and eagle3 checkpoints
  - Positive Tests (expected to PASS):
    - Adapter on a trimmed real dflash config fixture yields: architectures
      accepted (`DFlashDraftModel`), vendored `block_size == 8 - 1 == 7`,
      `target_layer_ids == [i-1 for i in aux_hidden_state_layer_ids]`,
      `draft_vocab_size == 32000`, markov/confidence disabled.
    - Adapter on a trimmed real eagle3 config fixture yields: pinned aux layer
      ids computed with the same formula as vLLM's
      `get_eagle3_default_aux_hidden_state_layers` for the target and
      recorded in the adapted config; `draft_vocab_size == 32000`.
    - d2t offset semantics round-trip: `target_id = draft_idx + d2t[draft_idx]`
      and the derived target→draft inverse map agree on membership with `t2d`.
  - Negative Tests (expected to FAIL / hard-error):
    - dflash config with `confidence_loss_alpha != 0` in training options →
      ValueError.
    - Markov-bearing checkpoint (markov_rank > 0) under `algo=dflash` →
      ValueError.
    - Unknown speculators `algorithm` or missing `block_size`/`mask_token_id`
      (dflash) → ValueError naming the checkpoint.
- AC-2: `algo=dflash` trains via DSparkRuntime on dtensor v2
  - Positive Tests:
    - `policy.draft.algo: dflash` routes to DSparkRuntime; a training step on
      the 1n8g recipe computes finite CE+TV draft loss with confidence loss
      exactly 0 and no markov bias applied.
    - The dflash alignment verification script
      (`tools/draft_verification/verify_dflash_alignment.py`) confirms
      label/mask/block alignment against the speculators dflash utilities
      (`select_anchors` / `get_base_indices_for_anchored_blocks`); anchor
      sampling policy divergence is documented, not replicated.
  - Negative Tests:
    - `algo=dflash` on the megatron backend → ValueError (unchanged
      megatron guard: eagle3-only).
    - Existing dspark recipes remain bit-identical in behavior (dspark unit
      tests and existing checkpoint-meta tests still pass).
- AC-3: `algo=eagle3` trains via Eagle3Runtime on dtensor v2 with TTT
  - Positive Tests:
    - Vendored eagle3 forward+loss matches the original speculators
      `core.py` on identical random inputs for ttt_steps=3, including
      `prev_correct` masking, teacher-forced input shifts, and padded B>1
      flattening (`tools/draft_verification/verify_eagle3_parity.py`).
    - Teacher-forced shifts never cross flattened document boundaries
      (parity script includes a two-document case).
    - Loss uses NeMo-RL normalization: DP-group denominator reduction and
      microbatch-slot scaling (finite, gbs/mbs-independent gradient scale).
  - Negative Tests:
    - `algo=eagle3` on dtensor v2 without `_v2: true` → ValueError (same
      backend guard as dspark).
    - `ttt_steps` inconsistent with `num_speculative_tokens` → warning
      emitted (not silent).
- AC-4: vLLM refit works for all three methods
  - Positive Tests:
    - Per-method refit manifest unit tests: expected/skipped key derivation
      against small fake drafter modules for dspark, dflash, and eagle3.
    - `tools/draft_verification/vllm_refit_preflight.py` runs green in the
      serving container: both drafters load, trainer-streamed keys match the
      expected sets, integer/bool buffers keep their dtype, and no
      embed/lm_head pointer aliasing exists between draft and target after
      the sharing disable.
  - Negative Tests:
    - A manifest missing a required draft key → existing hard error (uniform
      across the three methods).
    - Eagle path with module sharing left enabled → preflight aliasing
      assertion fails (this is the guard the disable exists for).
- AC-5: Checkpoint metadata is versioned and per-algo
  - Positive Tests:
    - Save/load round-trip for dflash and eagle3 records `meta_version`,
      `algo`, per-algo fields (dflash: block_size, mask_token_id,
      target_layer_ids, draft_vocab_size; eagle3: aux_layer_ids, ttt_steps,
      draft_vocab_size), trainable-embed flag, and optimizer layout.
    - Legacy dspark meta (no `algo`, no `meta_version`) still loads (treated
      as dspark/v0).
  - Negative Tests:
    - Resuming with mismatched per-algo fields (e.g. different aux_layer_ids)
      → ValueError refusing the resume.
- AC-6: 1n8g smoke runs pass the approved criteria (both recipes)
  - Positive Tests:
    - ~5–10 GRPO steps complete without crashing; draft loss trends down
      (dflash: CE+TV; eagle3: per-step soft-CE sum).
    - vLLM `spec_acceptance_length` is healthy from step 0 and does not
      collapse across refits (no hard threshold per user decision; offline
      scale anchors: dflash 3.53, eagle3 2.39@k3 on DAPO/thinking).
    - The dspark-parity metric set is emitted (draft loss components,
      per-position acceptance, draft_grad_norm finite).
  - Negative Tests:
    - Config validation tests reject recipes with hidden call-site defaults
      for the new draft options (defaults live on the BaseModel / exemplar
      YAML only).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation delivers: the draft-runtime protocol generalization
(worker/loss-wrapper/`setup.py` routing with backward-compatible dspark
aliases), the dflash adapter + validation + recipe, the vendored eagle3 model
+ Eagle3Runtime + recipe, vLLM refit generalization for the three methods
with the eagle-path module-sharing disable, versioned per-algo checkpoint
metadata, the three checked-in verification scripts under
`tools/draft_verification/`, per-method refit manifest unit tests, adapter
unit tests, exemplar-YAML documentation for the new options, and both 1n8g
smoke runs green. No megatron changes, no nightly wiring, no sequence-packing
/ LoRA / multi-node support.

### Lower Bound (Minimum Acceptable Scope)

The implementation delivers: `algo=dflash` and `algo=eagle3` training end to
end on dtensor v2 with the two recipes, correct adapter mappings (block−1,
aux-id −1, d2t offsets), eagle3 TTT loss verified against the speculators
oracle, refit for all three methods with the eagle sharing disable and
manifest hard-errors, per-algo checkpoint meta with legacy-dspark
acceptance, the verification scripts runnable, and both smoke runs meeting
AC-6. Unit-test coverage may be minimal but must cover AC-1 mappings and
AC-4 manifest derivation.

### Allowed Choices

- Can use: the existing DSparkRuntime/loss/capture/CP-gather machinery for
  dflash; vendored speculators eagle3 code with documented local
  modifications; pydantic `BaseModel(extra="allow")` for the eagle3 options
  schema; an env-var or unconditional-under-co-training mechanism for the
  eagle module-sharing disable (mirroring the dspark precedent); either
  `tools/draft_verification/` or `tests/` placement for oracle scripts (not
  CI-wired).
- Cannot use: changes to the megatron eagle3 path; new `TypedDict` config
  classes; silent fallbacks for adapter/validation failures; replicating
  speculators' anchor-sampling policy for dflash (the RL-side random-anchor
  scheme is fixed by design); hard acceptance-rate thresholds in smoke
  gating (user decision).
- Fixed by design (deterministic): checkpoint/method pairings
  (RedHat dflash → `method: dflash`, spec tokens 7; RedHat eagle3 →
  `method: eagle3`, spec tokens 3, ttt_steps 3); teacher targets from stashed
  policy logits; `train_embed_and_head: true`; recipes inherit
  `grpo-qwen3-8b-1n8g-automodel.yaml`.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. Generalize naming/routing first so both new algos land on a stable
   surface: `draft_runtime` protocol (`attach_capture / begin_global_batch /
   stash_teacher_logits / compute_loss`), algo routing in
   `nemo_rl/models/automodel/setup.py`, guard updates in
   `nemo_rl/models/policy/lm_policy.py` (dtensor v2 accepts
   eagle3/dspark/dflash; megatron stays eagle3-only).
2. DFlash rides DSparkRuntime: extend
   `load_dspark_draft_hf_config`/`_adapt_speculators_dspark_config` to accept
   `DFlashDraftModel` and map `block_size → block_size − 1`; validate the
   mapping with the alignment script against the speculators dflash utils
   before trusting a smoke run.
3. EAGLE3: vendor `speculators/models/eagle3/core.py` (+ helpers) with a
   provenance header; replace `verifier_last_hidden_states` with a
   `teacher_logits` argument (runtime stashes policy raw logits, applies the
   d2t offset gather, replicates `compute_metrics` per-step slicing exactly);
   flatten B>1 padded microbatches into one packed row with
   `document_ids = batch index` and per-sequence position ids; wrap in
   `Eagle3Runtime` with DP-denominator + microbatch-slot normalization.
4. Refit: parameterize the dspark-specific validation in
   `nemo_rl/models/generation/vllm/vllm_backend.py` by method; read the
   eagle3 key-naming rules off the serving container's
   `qwen3_eagle3.py::load_weights`; extend the module-sharing disable (dspark
   env precedent) to `vllm/v1/worker/gpu/spec_decode/eagle/utils.py::_should_share`.
5. Run the three verification scripts in the serving container, then the two
   1n8g smoke runs.

### Relevant References

- `nemo_rl/models/automodel/draft/integration.py` — speculators adapter,
  DSparkRuntime, capture (ModuleDict-aware, −1 embedding), CP gathers,
  checkpoint meta.
- `nemo_rl/models/automodel/draft/draft_qwen3.py` — vendored dspark/dflash
  model (reduced vocab, d2t offsets, t2d inverse map).
- `nemo_rl/models/automodel/setup.py`, `nemo_rl/models/policy/lm_policy.py`,
  `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py`,
  `nemo_rl/algorithms/loss/wrapper.py` — routing, guards, worker wiring,
  loss wrapper.
- `nemo_rl/models/generation/vllm/vllm_backend.py`,
  `nemo_rl/models/generation/vllm/vllm_worker.py` — refit validation,
  module-sharing disable precedent (`DSPARK_DISABLE_DRAFT_MODULE_SHARING_ENV`).
- `/lustre/fsw/portfolios/coreai/users/yuekaiz/speculative/speculators` —
  oracle: `src/speculators/models/eagle3/core.py` (TTT loop),
  `src/speculators/models/dflash/utils.py` (block/anchor semantics).
- Serving container vLLM 0.25.1:
  `vllm/model_executor/models/qwen3_eagle3.py` (loader naming, aux sizing),
  `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py` (aux id ±1
  conversion), `vllm/v1/worker/gpu/spec_decode/eagle/utils.py`
  (module sharing).
- `nemo_rl/models/megatron/draft/` — megatron eagle3 reference (unchanged).
- `examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-automodel{,-dspark}.yaml`
  — recipe base and knob alignment source.

## Dependencies and Sequence

### Milestones
1. Milestone 1 — Runtime generalization: draft_runtime protocol, algo
   routing (dspark|dflash → DSparkRuntime, eagle3 → Eagle3Runtime on dtensor
   v2), guard updates, versioned per-algo checkpoint metadata with legacy
   dspark acceptance. Everything later depends on this surface.
2. Milestone 2 — DFlash enablement (depends on M1): adapter extensions +
   validation, dflash alignment verification script, refit method set gains
   dflash, recipe + adapter/manifest unit tests.
3. Milestone 3 — EAGLE3 enablement (depends on M1; independent of M2):
   vendored eagle3 model + parity script, Eagle3Runtime + loss
   normalization, `Eagle3DraftOptions` BaseModel + exemplar docs, eagle3
   refit keys + module-sharing disable, recipe + unit tests.
4. Milestone 4 — Verification and smoke (depends on M2 and M3): refit
   preflight script green in the serving container, both 1n8g smoke runs
   meeting AC-6, results captured for the landing commit.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Generalize worker/loss-wrapper/setup routing to a draft_runtime protocol with algo dispatch and backward-compatible dspark aliases; update lm_policy guards | AC-2, AC-3 | coding | - |
| task2 | Versioned per-algo checkpoint metadata (meta_version, algo, per-algo fields, trainable-embed flag) with legacy dspark meta acceptance + unit tests | AC-5 | coding | task1 |
| task3 | Adapter: accept DFlashDraftModel, block_size−1 mapping, dflash validation rules (confidence=0, no markov) + unit tests on trimmed real fixtures | AC-1, AC-2 | coding | task1 |
| task4 | `tools/draft_verification/verify_dflash_alignment.py` vs speculators dflash utils; run and record result | AC-2 | coding | task3 |
| task5 | Vendor speculators eagle3 model (provenance header, copyright) with teacher_logits substitution and B>1 packed-row flattening | AC-3 | coding | task1 |
| task6 | `tools/draft_verification/verify_eagle3_parity.py` (forward+loss parity incl. prev_correct, shifts, two-document flattening); run and record result | AC-3 | coding | task5 |
| task7 | Eagle3Runtime (capture pinned aux ids −1, TTT loss with DP/microbatch-slot normalization) + Eagle3DraftOptions BaseModel + exemplar YAML docs | AC-3 | coding | task5 |
| task8 | Refit generalization: per-method expected/skipped key rules, eagle module-sharing disable, per-method manifest unit tests | AC-4 | coding | task1 |
| task9 | `tools/draft_verification/vllm_refit_preflight.py`; run green in serving container for both drafters | AC-4 | coding | task8 |
| task10 | Two recipes (dflash spec7/FLASH_ATTN, eagle3 spec3) with experimental/qwen3-only headers; config validation passes | AC-6 | coding | task3, task7 |
| task11 | Review the assembled diff for convention violations (config defaults at call sites, silent fallbacks, naming leaks of plan terminology) before smoke | AC-1..AC-6 | analyze | task2, task4, task6, task9, task10 |
| task12 | 1n8g smoke runs for both recipes; verify AC-6 metrics; capture logs for landing commit | AC-6 | coding | task11 |

## Claude-Codex Deliberation

### Agreements

- DFlash-as-degenerate-DSpark reuse, block_size−1 mapping with oracle gating,
  eagle3 aux-id pinning, teacher targets from stashed policy raw logits,
  B>1 packed-row flattening with boundary-safe shifts, refit preflight before
  smoke, eagle-path module-sharing disable (verified in container source),
  experimental scope without nightly wiring, TP/CP as loud-fail (not
  acceptance gates), documented divergence on dflash anchor-sampling policy.

### Resolved Disagreements

- **Eagle3 options schema**: Codex objected to a new `TypedDict`
  (repo convention: new user-facing config is `pydantic.BaseModel`). Resolved
  in Claude revision: `Eagle3DraftOptions` is a `BaseModel(extra="allow")`
  with defaults on fields, validated at runtime via `model_validate`;
  exemplar YAML documents the defaults. No new TypedDicts.
- **Verification reproducibility**: Codex required checked-in artifacts
  instead of manual claims. Resolved: three scripts under
  `tools/draft_verification/` + smoke logs referenced in the landing commit.
- **Checkpoint metadata**: Codex required per-algo versioned metadata.
  Resolved: meta_version + algo + per-algo fields with legacy dspark
  acceptance (task2).

### Convergence Status

- Final Status: `converged` (2 rounds; round 2 returned no DISAGREE and no
  REQUIRED_CHANGES).
- Execution-gated assumptions carried into implementation (not user
  decisions): container vLLM 0.25.1 eagle sharing behavior and speculators
  oracle parity must run green in the serving container before smoke results
  count (task6, task9 are the gates).

## Pending User Decisions

- None. All first-pass `QUESTIONS_FOR_USER` items were resolved during
  design/convergence: dflash anchor-policy divergence (documented, accepted),
  aux-id pinning (adopted), TP/CP gates (loud-fail only, per approved spec),
  nightly coverage (experimental only, per scope decision), eagle3
  embed-sharing handling (sharing disable + preflight aliasing assertion).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

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

# DFlash + EAGLE3 Qwen3-8B Speculative-Decoding RL Co-Training (Automodel Path)

Date: 2026-08-19 · Branch: `dflash` · Status: approved design

## Goal and scope

Extend the DTensor-v2 (automodel) draft co-training path — today dspark-only —
to two more drafter families, each with a runnable 1n8g GRPO recipe:

- **DFlash**: `RedHatAI/Qwen3-8B-speculator.dflash` (speculators format,
  block_size 8, 7 speculative tokens, reduced 32000-token draft vocab).
- **EAGLE3**: `RedHatAI/Qwen3-8B-speculator.eagle3` (speculators format,
  1 decoder layer, 3 speculative tokens, reduced 32000-token draft vocab),
  trained with speculators-style **TTT unrolling** (user decision: TTT, not
  the megatron backend's single-step distillation).

Target model: `Qwen/Qwen3-8B`. Both recipes inherit
`grpo-qwen3-8b-1n8g-automodel.yaml` (non-thinking, 2048-token generation) and
keep training knobs aligned with `grpo-qwen3-8b-1n8g-automodel-dspark.yaml`
for cross-method comparison (user decision: "run first, align knobs").
`train_embed_and_head: true` for both (user decision).

Out of scope: megatron-backend changes (its eagle3 path is untouched),
sequence packing, LoRA, multi-node recipes.

## Key facts grounding the design

- The vendored `Qwen3DSparkModel` with `markov_rank=0` and no confidence head
  **is** DFlash: `deepseek-ai/dflash_qwen3_8b_block7` ships exactly that
  (declared arch `Qwen3DSparkModel`, no markov/confidence tensors) and decodes
  at 3.26 accept-length on DAPO via vLLM `method=dspark`.
- The RedHat dflash checkpoint's weights are byte-layout-identical to the
  vendored model minus markov/confidence (embed 151936×4096, fc 5H→H,
  lm_head 32000×4096, `d2t` offsets int64, `t2d` bool).
- speculators `block_size` counts the anchor slot: dflash block 8 = anchor +
  7 predicted positions; the vendored trainer's `block_size` counts predicted
  positions only (deepseek block7 = 7 predictions).
- speculators/vLLM `aux_hidden_state_layer_ids` use vLLM capture indexing
  (id j = output of decoder layer j−1; 0 = embedding); the vendored capture
  uses output-of-layer-i with −1 for embeddings. Confirmed in vLLM source
  (`_maybe_add_hidden_state(aux, idx + 1, ...)`) and by the offset-mapping
  utility (`[i + 1 for i in target_layer_ids]`).
- `d2t` stores offsets: `target_id = draft_idx + d2t[draft_idx]` (matches
  vLLM's `arange + d2t`).
- The RedHat eagle3 checkpoint pins no aux layer ids; vLLM falls back to the
  target model's `get_eagle3_default_aux_hidden_state_layers()`. The trainer
  capture must mirror that same selection (with the ±1 conversion).
- speculators eagle3 training (`speculators/models/eagle3/core.py`): TTT loop
  (default 3 steps), per-step soft-CE against
  `verifier_lm_head(verifier_norm(verifier_last_hidden))` restricted to the
  draft vocab, `prev_correct` masking (a position contributes to step-k loss
  only if argmax was correct on all previous steps), teacher-forced input ids
  shifted by 1+step, KV cache accumulated across steps, document-id attention
  masks. Defaults: `ttt_steps=3`, `ttt_step_loss_decay=1.0`.
- vLLM 0.25.1 (container) serves both checkpoints already: measured on DAPO
  math (thinking, temp 0.6) — dflash 3.535 accept-length / 80.6% pos-1,
  eagle3 2.385 / 68.8% (k=3); both README tables reproduced exactly.

## Architecture

```
nemo_rl/models/automodel/draft/
├── draft_qwen3.py, loss.py, markov_head.py   # existing, shared by dspark+dflash
├── eagle3/                                    # NEW: vendored from speculators
│   └── core.py (+ attention/config as needed)
└── integration.py
    ├── speculators adapter (extended)
    ├── DSparkRuntime          # reused by algo=dspark and algo=dflash
    └── Eagle3Runtime          # NEW, same runtime protocol
```

Routing: `policy.draft.algo ∈ {eagle3, dspark, dflash}`. On dtensor v2,
`dspark|dflash → DSparkRuntime`, `eagle3 → Eagle3Runtime`; on megatron,
`eagle3` keeps its existing path unchanged. Worker and loss-wrapper
`dspark_runtime` references generalize to a `draft_runtime` protocol
(`attach_capture / begin_global_batch / stash_teacher_logits / compute_loss`),
keeping backward-compatible aliases where megatron code touches them.

### DFlash via DSparkRuntime

- Adapter accepts `DFlashDraftModel` architectures; missing markov/confidence
  fields default to disabled (already the adapter's behavior).
- Adapter maps speculators `block_size` → vendored `block_size − 1`
  (anchor-slot convention). This mapping is verified against the speculators
  source (`select_anchors` / `get_base_indices_for_anchored_blocks`) before
  landing (see Verification).
- `confidence_loss_alpha` must be 0 for dflash (validate hard-errors
  otherwise); markov is absent so the Markov bias path is inert.
- Everything else (reduced vocab d2t/t2d, capture, CP gathers, FSDP over
  dp_cp, named optimizer groups, DCP draft checkpoint, refit streaming with
  integer-buffer dtype preservation) is reused as-is.

### EAGLE3 via Eagle3Runtime

- Vendor `speculators/models/eagle3/core.py` (+ its attention/config helpers)
  into `nemo_rl/models/automodel/draft/eagle3/`, with local modifications in
  the header-documented style of `draft_qwen3.py`:
  - Replace `verifier_last_hidden_states` + frozen verifier-head copies with a
    `teacher_logits` argument: the runtime stashes the policy's raw logits
    (dspark pattern) and maps them into draft-vocab order via the d2t offsets.
    Mathematically identical to the frozen-head construction and stays exact
    as the policy drifts during RL.
  - Support B>1 padded microbatches by flattening the batch into one packed
    row with `document_ids = batch index` and per-sequence `position_ids`,
    reusing the vendored document-mask machinery. Padded positions are
    excluded via loss_mask.
- Capture: 3 aux layer outputs (layer ids = vLLM default selection converted
  by −1) concatenated to [B, S, 3H]; input embeddings are recomputed by the
  draft's own `embed_tokens` (as in the vendored forward), so no embedding
  capture is required.
- Loss: per-TTT-step soft-CE with `prev_correct` masking and
  `ttt_step_loss_decay` weighting, summed; scaled by the shared
  microbatch-slot normalization (`begin_global_batch`) like dspark.
- CP: reuse the dspark gather path (hiddens/teacher/input_ids restored to full
  sequences via the load-balanced allgather). 1n8g recipes do not exercise CP;
  the code path exists but is not an acceptance gate.

## vLLM refit and configuration surface

Refit generalization in `vllm_backend.py`:

- Speculator-owning checks and draft-stream validation activate for
  `method ∈ {dspark, dflash, eagle3}`.
- Expected-key derivation stays "enumerate drafter `named_parameters()` and
  invert the loader's naming rules", with the naming rules dispatched per
  drafter class (dspark/dflash share rules; eagle3's are read off
  `qwen3_eagle3.py::load_weights` during implementation).
- Skipped-key sets parameterized per method (dflash: `t2d`; eagle3: `t2d`
  plus whatever the eagle3 drafter never loads — determined at
  implementation, e.g. whether it owns `embed_tokens` or shares the
  target's).
- Trainer-side streaming is unchanged (dtype preservation already landed).

Config schema (`policy.draft`):

- `algo` gains `dflash`; guard updates: dtensor v2 accepts all three algos,
  megatron stays eagle3-only. The dspark-specific unsupported-set
  (sequence_parallel, LoRA, packing rejected; TP/CP/AC allowed) applies to all
  three on dtensor v2.
- dflash reuses the existing `policy.draft.dspark` options block unchanged
  (fields are a superset; `confidence_loss_alpha: 0.0` enforced).
- eagle3 adds a `policy.draft.eagle3` options block (TypedDict following the
  `DSparkDraftOptions` precedent): `learning_rate`, `ttt_steps: 3`,
  `ttt_step_loss_decay: 1.0`, `train_embed_and_head: true`. Defaults +
  documentation go into the exemplar `examples/configs/grpo_math_1B.yaml`
  (and `tests/unit/reference_configs/` if covered by config tests).
- Worker's `num_speculative_tokens` consistency warning dispatches per algo:
  dflash expects block_size−1 (=7), eagle3 expects ttt_steps (=3).

Recipes (both inherit `grpo-qwen3-8b-1n8g-automodel.yaml`):

```yaml
# grpo-qwen3-8b-1n8g-automodel-dflash.yaml
policy:
  draft:
    enabled: true
    model_name: RedHatAI/Qwen3-8B-speculator.dflash
    algo: dflash
    loss_weight: 1.0
    dspark: {num_anchors: 32, learning_rate: 1.0e-4, ce_loss_alpha: 0.1,
             l1_loss_alpha: 0.9, confidence_loss_alpha: 0.0,
             loss_decay_gamma: 4.0, train_embed_and_head: true}
  generation:
    vllm_kwargs:
      speculative_config: {method: dflash, model: <model_name>,
                           num_speculative_tokens: 7,
                           attention_backend: FLASH_ATTN}

# grpo-qwen3-8b-1n8g-automodel-eagle3.yaml
policy:
  draft:
    enabled: true
    model_name: RedHatAI/Qwen3-8B-speculator.eagle3
    algo: eagle3
    loss_weight: 1.0
    eagle3: {learning_rate: 1.0e-4, ttt_steps: 3, ttt_step_loss_decay: 1.0,
             train_embed_and_head: true}
  generation:
    vllm_kwargs:
      speculative_config: {method: eagle3, model: <model_name>,
                           num_speculative_tokens: 3}
```

## Error handling (loud, never silent)

- Adapter: unknown speculators `algorithm`, missing required fields
  (block_size/mask_token_id for dflash; aux-layer derivation failure for
  eagle3) → hard error naming the checkpoint.
- `algo=dflash` with `confidence_loss_alpha != 0` or a markov-bearing
  checkpoint → hard error (no silent degradation).
- Refit manifest mismatch → existing hard-error path, uniform across methods.
- `ttt_steps` vs `num_speculative_tokens` inconsistency → warning (same style
  as the dspark block-size warning).

## Verification plan

1. **Unit tests** (`tests/unit/models/automodel/`): adapter mappings
   (dflash/eagle3 configs → vendored fields: −1 layer shift, block_size−1,
   d2t offset semantics) using trimmed real-config fixtures; d2t/t2d
   round-trip; checkpoint-meta compatibility extensions.
2. **Oracle alignment (one-off scripts, not CI)**: vendored eagle3 forward vs
   the original speculators `core.py` on identical random inputs (speculators
   source available locally as the oracle); dflash block/label alignment
   checked against `select_anchors` / `get_base_indices_for_anchored_blocks`.
   These exist specifically to de-risk the two ±1 conventions (block-size
   anchor slot, aux layer ids) that caused the 35B feature-misalignment
   incident.
3. **1n8g smoke runs** (acceptance criteria, user-approved):
   - trains ~5–10 steps without crashing; draft loss trends down
     (dflash: CE+TV; eagle3: per-step soft-CE);
   - vLLM `spec_acceptance_length` healthy from step 0 and does not collapse
     across refits (offline anchors for scale: dflash 3.53, eagle3 2.39@k3 on
     DAPO/thinking; recipes are non-thinking + temp 1.0, so no hard
     threshold — "not collapsed, stable-or-rising");
   - the full dspark-parity metric set is emitted (draft loss components,
     acceptance, draft_grad_norm).

## Risk register

- **±1 conventions** (block-size anchor slot; eagle3 default aux layer ids):
  mitigated by the oracle checks above; both have precedent failures.
- **eagle3 refit key mapping**: drafter-class naming differences are read off
  the vLLM loader during implementation; the manifest validation hard-errors
  on any mismatch, so a wrong mapping cannot pass silently.
- **TTT memory**: 3-step unroll over full 4096-token microbatches with KV
  cache accumulation; at 1n8g with a 1-layer draft this is small
  (~3×S×H activations), but the smoke run confirms headroom.

## Key user decisions recorded

- Scope C: get both running first, knobs aligned to the dspark recipe for
  comparison.
- `train_embed_and_head: true` for both drafters.
- EAGLE3 uses speculators-style TTT training (not megatron single-step
  distillation); defaults ttt_steps=3, decay 1.0.
- Acceptance criteria: the three smoke-run conditions above (option A, no
  hard step-0 threshold).

--- Original Design Draft End ---
