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
