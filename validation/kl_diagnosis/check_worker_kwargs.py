"""Exercise the real worker's engine-argument assembly without loading a model."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs

from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from validation.kl_diagnosis.worker_setup import install_generation

register_omegaconf_resolvers()
model = MasterConfig(
    **OmegaConf.to_container(
        load_config("validation/kl_diagnosis/ep16_noprefix.yaml"), resolve=True
    )
)
cfg = copy.deepcopy(model.policy["generation"])
cfg["model_name"] = model.policy["model_name"]
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
cfg = configure_generation_config(cfg, tokenizer)
assert cfg["vllm_cfg"]["enable_prefix_caching"] is False
assert "enable_prefix_caching" not in cfg["vllm_kwargs"]
install_generation()
captured = []


class ArgumentsCaptured(Exception):
    pass


def capture_engine(kwargs):
    captured.append(copy.deepcopy(kwargs))
    raise ArgumentsCaptured


def worker(config):
    options = config["vllm_cfg"]
    return SimpleNamespace(
        cfg=config,
        model_name=config["model_name"],
        tensor_parallel_size=options["tensor_parallel_size"],
        pipeline_parallel_size=options["pipeline_parallel_size"],
        expert_parallel_size=options["expert_parallel_size"],
        enable_expert_parallel=options["expert_parallel_size"] > 1,
        gpu_memory_utilization=options["gpu_memory_utilization"],
        fraction_of_gpus=8.0,
        precision=options["precision"],
        _create_engine=capture_engine,
    )


# Reproduce the exact r12 error at the real dict-construction boundary.
broken = copy.deepcopy(cfg)
broken["vllm_kwargs"]["enable_prefix_caching"] = False
with patch.dict(os.environ, dict(os.environ)):
    try:
        BaseVllmGenerationWorker._load_model(worker(broken), list(range(8)), 42)
    except TypeError as error:
        assert "multiple values" in str(error) and "enable_prefix_caching" in str(error)
    else:
        raise AssertionError("The old kwargs placement must reproduce the duplicate")

# Keep model/engine creation out of this preflight; run the production assembly.
with patch.dict(os.environ, dict(os.environ)):
    try:
        BaseVllmGenerationWorker._load_model(worker(cfg), list(range(8)), 42)
    except ArgumentsCaptured:
        pass
    else:
        raise AssertionError("The real worker must reach engine construction")
assert len(captured) == 1
kwargs = captured[0]
assert kwargs["enable_prefix_caching"] is False
assert kwargs["tensor_parallel_size"] == 8
assert kwargs["max_num_batched_tokens"] == 2048
assert kwargs["mamba_ssm_cache_dtype"] == "float32"
assert kwargs["mamba_cache_dtype"] == "bfloat16"
engine_config = AsyncEngineArgs(**kwargs).create_engine_config()
assert engine_config.cache_config.enable_prefix_caching is False
assert engine_config.scheduler_config.max_num_batched_tokens == 2048
report = {
    "status": "passed",
    "scope": "Real BaseVllmGenerationWorker._load_model up to _create_engine; actual AsyncEngineArgs config validation; no model or engine created",
    "old_kwargs_reproduced_duplicate_error": True,
    "engine_kwargs": {
        name: kwargs[name]
        for name in (
            "enable_prefix_caching",
            "tensor_parallel_size",
            "max_model_len",
            "max_num_batched_tokens",
            "mamba_cache_dtype",
            "mamba_ssm_cache_dtype",
            "load_format",
        )
    },
    "mamba_cache_mode": engine_config.cache_config.mamba_cache_mode,
}
(Path(os.environ["SC_LP_DIAG_DIR"]) / "worker-kwargs-preflight.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print("WORKER_KWARGS_PREFLIGHT", json.dumps(report), flush=True)
