"""Sample every refit payload tensor and every resident expert/router tensor."""

import os
from pathlib import Path

import torch

from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension


from validation.kl_diagnosis.sampling import sample_tensor


class ProbeExtension(VllmInternalWorkerExtension):
    def diagnostic_probe_status(self):
        return {"probe": "ProbeExtension", "rank": torch.distributed.get_rank()}

    def _diagnostic_gdn_layers(self):
        layers = [
            (name, module)
            for name, module in self.model_runner.model.named_modules()
            if getattr(
                getattr(module, "chunk_gated_delta_rule", None),
                "gdn_prefill_backend",
                None,
            )
            is not None
        ]
        assert len(layers) == 30, len(layers)
        return layers

    def diagnostic_execution_status(self):
        from vllm import envs
        from vllm.model_executor.determinism import batch_invariant

        layers = []
        for name, module in self._diagnostic_gdn_layers():
            cache = module.kv_cache
            layers.append(
                {
                    "name": name,
                    "prefill_backend": module.chunk_gated_delta_rule.gdn_prefill_backend,
                    "prefill_method": module.chunk_gated_delta_rule._forward_method.__name__,
                    "decode_kernel": module.gdn_decode_kernel,
                    "configured_state_dtypes": [
                        str(x) for x in module.get_state_dtype()
                    ],
                    "conv_cache_dtype": str(cache[0].dtype),
                    "ssm_cache_dtype": str(cache[1].dtype),
                    "a_log_dtype": str(module.A_log.dtype),
                    "dt_bias_dtype": str(module.dt_bias.dtype),
                }
            )
        return {
            "rank": torch.distributed.get_rank(),
            "cache_scheduler": {
                "enable_prefix_caching": self.cache_config.enable_prefix_caching,
                "mamba_cache_mode": self.cache_config.mamba_cache_mode,
                "enable_chunked_prefill": self.scheduler_config.enable_chunked_prefill,
                "max_num_batched_tokens": self.scheduler_config.max_num_batched_tokens,
            },
            "batch_invariance": {
                "env_enabled": envs.VLLM_BATCH_INVARIANT,
                "initialized": batch_invariant._batch_invariant_MODE,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cublaslt_workspace_size": os.environ.get("CUBLASLT_WORKSPACE_SIZE"),
                "fp32_matmul_precision": torch.backends.cuda.matmul.fp32_precision,
                "bf16_reduced_precision_reduction": (
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                ),
            },
            "layers": layers,
            "model_training": self.model_runner.model.training,
            "active_dropout_modules": [
                name
                for name, module in self.model_runner.model.named_modules()
                if isinstance(module, torch.nn.Dropout)
                and module.training
                and module.p > 0
            ],
        }

    def diagnostic_resident_snapshot(self, label):
        assert label in {"before", "after"}
        rank = torch.distributed.get_rank()
        torch.save(
            {
                name: sample_tensor(value)
                for name, value in self._get_named_parameters().items()
            },
            Path(os.environ["SC_LP_DIAG_DIR"]) / f"repeat-weights-{label}-{rank}.pt",
        )
        return {"rank": rank, "label": label}

    def diagnostic_prefill_backend(self, restore=False):
        """Use the image's official alternate prefill method; preserve all weights."""
        assert self.model_config.enforce_eager
        if restore:
            originals = getattr(self, "_diag_prefill_original", [])
            for module, method, backend, outer_backend in originals:
                operation = module.chunk_gated_delta_rule
                operation._forward_method = method
                operation.gdn_prefill_backend = backend
                module.gdn_prefill_backend = outer_backend
            self._diag_prefill_original = []
            return {
                "rank": torch.distributed.get_rank(),
                "restored_layers": len(originals),
                "triton_calls": dict(getattr(self, "_diag_prefill_calls", {})),
            }
        assert not getattr(self, "_diag_prefill_original", [])
        self._diag_prefill_original = []
        self._diag_prefill_calls = {}
        for name, module in self._diagnostic_gdn_layers():
            operation = module.chunk_gated_delta_rule
            assert operation.gdn_prefill_backend == "flashinfer"
            self._diag_prefill_original.append(
                (
                    module,
                    operation._forward_method,
                    operation.gdn_prefill_backend,
                    module.gdn_prefill_backend,
                )
            )
            self._diag_prefill_calls[name] = 0

            def counted_forward(
                *args, _method=operation.forward_native, _name=name, **kwargs
            ):
                self._diag_prefill_calls[_name] += 1
                if torch.distributed.get_rank() == 0 and not getattr(
                    self, "_diag_captured_gdn", False
                ):
                    tensors = {
                        key: value.detach().cpu().clone()
                        for key, value in kwargs.items()
                        if isinstance(value, torch.Tensor)
                    }
                    assert all(
                        key in tensors
                        for key in (
                            "q",
                            "k",
                            "v",
                            "g",
                            "beta",
                            "initial_state",
                            "cu_seqlens",
                        )
                    )
                    torch.save(
                        {
                            "layer": _name,
                            "tensors": tensors,
                            "use_qk_l2norm_in_kernel": kwargs[
                                "use_qk_l2norm_in_kernel"
                            ],
                        },
                        Path(os.environ["SC_LP_DIAG_DIR"]) / "gdn-prefill-inputs.pt",
                    )
                    self._diag_captured_gdn = True
                return _method(*args, **kwargs)

            operation._forward_method = counted_forward
            operation.gdn_prefill_backend = "triton"
            module.gdn_prefill_backend = "triton"
        return {
            "rank": torch.distributed.get_rank(),
            "triton_layers": len(self._diag_prefill_original),
        }

    @torch.no_grad()
    def diagnostic_ssm_precision(self, restore=False):
        if restore:
            for name, value in self._diag_ssm_original.items():
                self._get_named_parameters()[name].copy_(value)
            return {"restored": len(self._diag_ssm_original)}
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        root = Path(os.environ["SC_LP_DIAG_DIR"])
        source = torch.load(
            root / "ssm-source-0.pt", map_location="cpu", weights_only=False
        )
        # A_log is explicitly FP32 in this image. dt_bias uses the model's
        # default dtype; do not assume its destination has the same contract.
        source = {
            name: value for name, value in source.items() if name.endswith(".A_log")
        }
        assert len(source) == 30, len(source)
        rank, world = (
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
        )
        resident = self._get_named_parameters()
        self._diag_ssm_original = {}
        for hf_name, full in source.items():
            suffix = "layers." + hf_name.split("layers.", 1)[1]
            names = [name for name in resident if name.endswith(suffix)]
            assert len(names) == 1, (hf_name, names)
            name = names[0]
            value = resident[name]
            assert value.dtype == torch.float32, (name, value.dtype)
            expected = full.chunk(world)[rank].to(value.device)
            assert expected.shape == value.shape, (name, expected.shape, value.shape)
            torch.testing.assert_close(
                value, expected.bfloat16().float(), rtol=0, atol=0
            )
            self._diag_ssm_original[name] = value.detach().clone()
            value.copy_(expected)
        return {"patched": len(self._diag_ssm_original), "rank": rank}

    def update_weights_from_collective(self, *args, **kwargs):
        self._diag_refit = getattr(self, "_diag_refit", -1) + 1
        self._diag_incoming = {}
        result = super().update_weights_from_collective(*args, **kwargs)
        rank = torch.distributed.get_rank()
        root = Path(os.environ["SC_LP_DIAG_DIR"])
        torch.save(self._diag_incoming, root / f"receiver-{self._diag_refit}-{rank}.pt")
        torch.save(
            {
                name: sample_tensor(value)
                for name, value in self._get_named_parameters().items()
            },
            root / f"resident-{self._diag_refit}-{rank}.pt",
        )
        return result

    def _load_weights(self, weights):
        weights = list(weights)
        self._diag_incoming.update(
            {name: sample_tensor(value) for name, value in weights}
        )
        return super()._load_weights(weights)
