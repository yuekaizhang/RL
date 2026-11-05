"""Inspect the real Qwen3.5 provider without allocating its model weights."""

import dataclasses
import json
from pathlib import Path

import megatron.bridge
import megatron.core
from megatron.bridge import AutoBridge
from omegaconf import OmegaConf
from transformers import AutoConfig

from nemo_rl.models.megatron.setup import (
    _apply_moe_config,
    _apply_multimodal_config,
    _apply_parallelism_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

register_omegaconf_resolvers()
config = OmegaConf.to_container(load_config("validation/mcore/la1.yaml"), resolve=True)
policy = config["policy"]
hf_config = AutoConfig.from_pretrained(policy["model_name"], trust_remote_code=True)
for key, value in policy["hf_config_overrides"]["text_config"].items():
    setattr(hf_config.text_config, key, value)
bridge = AutoBridge.from_hf_config(hf_config)
provider = bridge.to_megatron_provider(load_weights=False)
provider = dataclasses.replace(provider, **policy["megatron_cfg"]["model_overrides"])
_apply_parallelism_config(provider, policy)
_apply_multimodal_config(provider, policy)
_apply_moe_config(provider, policy)
assert provider.expert_model_parallel_size == 16
assert provider.tensor_model_parallel_size == provider.context_parallel_size == 1
assert provider.freeze_vision_model and provider.freeze_vision_projection
assert provider.moe_aux_loss_coeff == 0
assert provider.moe_router_pre_softmax is False
result = {
    "status": "passed",
    "bridge_source": megatron.bridge.__file__,
    "mcore_source": megatron.core.__file__,
    "provider": type(provider).__name__,
    "fields": {
        name: str(getattr(provider, name))
        for name in [
            "num_layers",
            "hidden_size",
            "num_moe_experts",
            "moe_router_topk",
            "expert_model_parallel_size",
            "tensor_model_parallel_size",
            "context_parallel_size",
            "freeze_vision_model",
            "freeze_vision_projection",
            "moe_router_dtype",
            "moe_router_pre_softmax",
            "moe_aux_loss_coeff",
        ]
    },
}
Path("validation/mcore/provider-probe.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
