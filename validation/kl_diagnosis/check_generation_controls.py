"""Check the pinned GDN backend and scoring controls before loading full models."""

import json
import os
from pathlib import Path

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")

from vllm import SamplingParams, envs
from vllm.config import CacheConfig, SchedulerConfig
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.selector import get_mamba_attn_backend

assert not envs.VLLM_BATCH_INVARIANT, "The pinned GDN backend rejects this mode"
backend = get_mamba_attn_backend(MambaAttentionBackendEnum.GDN_ATTN)
assert backend.get_name() == "GDN_ATTN"
assert not backend.supports_batch_invariance()
sampling = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
assert sampling.skip_reading_prefix_cache is True
cache = CacheConfig(
    enable_prefix_caching=False,
    mamba_cache_dtype="bfloat16",
    mamba_ssm_cache_dtype="float32",
)
scheduler = SchedulerConfig(
    max_model_len=4096,
    is_encoder_decoder=False,
    max_num_batched_tokens=2048,
    enable_chunked_prefill=True,
)
report = {
    "status": "passed",
    "scope": "Pinned backend selector, scoring defaults and cache/scheduler config; actual runtime is checked again on all eight workers",
    "gdn_backend": backend.get_name(),
    "batch_invariance_supported": backend.supports_batch_invariance(),
    "prompt_logprobs_skip_prefix_reads": sampling.skip_reading_prefix_cache,
    "enable_prefix_caching": cache.enable_prefix_caching,
    "ssm_dtype": cache.mamba_ssm_cache_dtype,
    "conv_dtype": cache.mamba_cache_dtype,
    "chunk_budget": scheduler.max_num_batched_tokens,
}
(Path(os.environ["SC_LP_DIAG_DIR"]) / "generation-controls-preflight.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print("GENERATION_CONTROLS_PREFLIGHT", json.dumps(report), flush=True)
