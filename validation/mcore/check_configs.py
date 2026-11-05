"""Validate the paired configurations and export the explicit matching knobs."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def load(name):
    config = OmegaConf.to_container(
        load_config(f"validation/mcore/{name}.yaml"), resolve=True
    )
    validate_single_controller_config(MasterConfig(**config))
    return config


def main():
    register_omegaconf_resolvers()
    cases = [load("la0"), load("la1")]
    for lag, config in enumerate(cases):
        assert config["async_rl"]["sampler"]["max_lookahead_versions"] == lag
        assert config["policy"]["train_global_batch_size"] == 32
        assert config["grpo"]["num_prompts_per_step"] == 8
        assert config["grpo"]["num_generations_per_prompt"] == 4
        assert config["policy"]["megatron_cfg"]["expert_model_parallel_size"] == 16
        assert config["policy"]["megatron_cfg"]["freeze_moe_router"] is False
        assert config["async_rl"]["max_buffered_rollouts"] == 16
        assert config["policy"]["generation"]["vllm_cfg"]["tensor_parallel_size"] == 8
        assert config["grpo"]["max_num_steps"] == 300
        assert config["loss_fn"]["truncated_importance_sampling_ratio"] == 2
        assert config["policy"]["megatron_cfg"]["optimizer"]["weight_decay"] == 0.1
    normalized = []
    for config in cases:
        comparable = json.loads(json.dumps(config))
        comparable["async_rl"]["sampler"]["max_lookahead_versions"] = None
        del comparable["checkpointing"]["checkpoint_dir"]
        del comparable["logger"]["log_dir"]
        del comparable["logger"]["wandb"]["id"]
        del comparable["logger"]["wandb"]["name"]
        normalized.append(comparable)
    assert normalized[0] == normalized[1], (
        "Paired configs differ beyond lookahead and output identity"
    )
    policy = cases[0]["policy"]
    manifest = {
        "status": "passed",
        "target_steps": 300,
        "only_case_differences": [
            "lookahead",
            "checkpoint/log directory",
            "W&B id/name",
        ],
        "model": policy["model_name"],
        "cluster": cases[0]["cluster"],
        "global_batch": policy["train_global_batch_size"],
        "prompts": 8,
        "responses": 4,
        "train_parallelism": {
            key: policy["megatron_cfg"][key]
            for key in [
                "tensor_model_parallel_size",
                "context_parallel_size",
                "pipeline_model_parallel_size",
                "expert_model_parallel_size",
            ]
        },
        "generation_tp": policy["generation"]["vllm_cfg"]["tensor_parallel_size"],
        "logprob_chunk_size": policy["logprob_chunk_size"],
        "allocator_env": policy["megatron_cfg"]["env_vars"],
        "optimizer": policy["megatron_cfg"]["optimizer"],
        "scheduler": policy["megatron_cfg"]["scheduler"],
        "router": {
            key: policy["megatron_cfg"][key]
            for key in [
                "freeze_moe_router",
                "moe_router_dtype",
                "moe_router_load_balancing_type",
                "model_overrides",
            ]
        },
        "loss": cases[0]["loss_fn"],
        "async_rl": cases[0]["async_rl"],
    }
    output = Path("validation/mcore/config-equivalence.json")
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key
                in [
                    "status",
                    "target_steps",
                    "only_case_differences",
                    "global_batch",
                    "train_parallelism",
                    "generation_tp",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
