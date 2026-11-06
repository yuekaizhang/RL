"""Exercise real Hydra composition and SC validation without creating actors."""

import json
import os
import sys
from pathlib import Path

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

register_omegaconf_resolvers()
root = Path(os.environ["SC_LP_DIAG_DIR"])
cfg = parse_hydra_overrides(
    load_config(sys.argv[1]),
    [
        f"logger.log_dir={root}/logs",
        f"checkpointing.checkpoint_dir={root}/ckpts",
    ],
)
model = MasterConfig(**OmegaConf.to_container(cfg, resolve=True))
validate_single_controller_config(model)
assert model.grpo.max_num_steps == 602
assert model.async_rl.sampler.max_lookahead_versions == 0
assert model.checkpointing["enabled"] is False
assert model.logger.wandb_enabled is False
if os.environ.get("SC_LP_CONTROL") == "prefix_cache_disabled":
    assert model.policy["generation"]["vllm_cfg"]["enable_prefix_caching"] is False
    assert model.policy["generation"]["vllm_kwargs"] == {
        "max_num_batched_tokens": 2048,
        "mamba_cache_dtype": "bfloat16",
        "mamba_ssm_cache_dtype": "float32",
    }
report = {
    "status": "passed",
    "config": sys.argv[1],
    "scope": "Real config inheritance, Hydra overrides, MasterConfig and SC validation; no actors/model loading",
    "vllm_kwargs": model.policy["generation"]["vllm_kwargs"],
    "enable_prefix_caching": model.policy["generation"]["vllm_cfg"].get(
        "enable_prefix_caching"
    ),
}
(root / "recipe-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
print("DIAGNOSTIC_RECIPE_PREFLIGHT", json.dumps(report), flush=True)
