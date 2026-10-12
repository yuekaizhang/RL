# Nemotron 3.5 Super VL — SWE Stage 1 (pivot) RL on the AutoModel backend

`swe1_pivot_automodel.yaml` runs the SWE **pivot** stage (PivotRL, single-step tool use
with argument comparison; see [`docs/guides/swe-rl-qwen3.md`](../../../docs/guides/swe-rl-qwen3.md))
for Nemotron 3.5 Super VL (`NemotronH_Omni_Reasoning_V3`, 120B-A12B) with:

- **Training**: AutoModel (DTensor/FSDP2) backend, EP4 + CP2, FusedAdam with FP32
  master weights, activation checkpointing. Text-only (no images).
- **Generation**: async GRPO, non-colocated stock vLLM 0.29 (TP4/EP4, one engine per
  4-GPU GB200 node), in-flight weight updates, NeMo Gym `vllm_model_for_training`.
- **Environment**: NeMo Gym `single_step_tool_use_with_argument_comparison` with the
  `swe_pivot` config; Nemotron tool-call format (`<tool_call><function=...>`) via the
  `qwen3_coder` tool parser and `<think>` blocks via the `nemotron_v3` reasoning parser.

## Data

```bash
# Nemotron-RL-Super-Training-Blends: swe1.jsonl (50,661 single-step pivot samples)
uvx --from huggingface-hub hf download nvidia/Nemotron-RL-Super-Training-Blends \
  --repo-type dataset --include swe1.jsonl --local-dir data_with_placeholders
# swe1 carries no dataset placeholders, so fill_placeholders.py is not needed for it.
mkdir -p data/swe1
head -n -100 data_with_placeholders/swe1.jsonl > data/swe1/train-split.jsonl
tail -n 100  data_with_placeholders/swe1.jsonl > data/swe1/val-split.jsonl
```

Prompt lengths (sampled): p50 ~22K, p90 ~43K, p99 ~76K, max ~83K tokens, which is
why the recipe keeps `max_total_sequence_length: 131072` and uses CP2 on the trainer.

## Launch

```bash
export HF_HOME=<shared cache> WANDB_API_KEY=<key>
uv run --frozen ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/nemotron-3.5-super-vl/swe1_pivot_automodel.yaml \
  data.train.data_path=/path/to/data/swe1/train-split.jsonl \
  data.validation.data_path=/path/to/data/swe1/val-split.jsonl
```

`cluster.num_nodes` is training + generation nodes;
`policy.generation.colocated.resources.num_nodes` is the generation share.

## Sizing (4-GPU GB200 nodes)

| Role | Nodes | Why |
|---|---:|---|
| Training | 8 | 32 GPUs: FSDP2 states ~38 GB/GPU for 121B params with FP32 masters + Adam; CP2 halves the per-GPU activation footprint of a 131K sequence (one micro-batch of one sequence, activation checkpointing). DP = 32 / CP2 = 16 divides the 512 rollouts per step. |
| Generation | 8 | One TP4/EP4 engine per node; 512 single-step rollouts per step with ~22K-token prompts are prefill-heavy, and async GRPO overlaps them with training. 4 generation nodes also work (slower rollouts). |
| **Total** | **16** | Minimum tried-and-true shape; 4 training + 2 generation nodes (6 total) may fit but leaves little headroom at 131K tokens. |

Trade-offs if memory is tight on the trainer: raise `context_parallel_size` to 4
(DP becomes 8, still divides 512), or lower `max_total_sequence_length` to 65536 and
accept that the longest ~10% of prompts get truncated by `cap_max_tokens_to_context`.

## Status

Recipe assembled from the Qwen3-30B SWE1 and Nemotron 3 Super SWE1 configs plus the
validated Super VL text DAPO settings; not yet run end to end. Items to confirm in the
first smoke: CP2 on the AutoModel Nemotron Omni model with the TE attention backend,
vLLM CUDA graphs (`enforce_eager: false`) for the hybrid Mamba model at 131K context,
and NeMo Gym venv availability in the container.
