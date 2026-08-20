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
