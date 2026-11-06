"""Install generation probes at the ordinary actor initialization boundary."""

import json
import os
import sys
from pathlib import Path

import ray

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


@ray.remote
class DiagnosticGenerationWorker(VllmAsyncGenerationWorkerImpl):  # pragma: no cover
    def __init__(self, *args, probe_only=False, **kwargs):
        from validation.kl_diagnosis.worker_setup import install_generation

        install_generation()
        assert "vllm_worker_async.VllmAsyncGenerationWorker" in sys.prefix, sys.prefix
        import vllm

        print(
            "DIAGNOSTIC_GENERATION_INTERPRETER",
            sys.executable,
            vllm.__file__,
            flush=True,
        )
        if probe_only:
            return
        super().__init__(*args, **kwargs)
        sampling = self._build_sampling_params(
            greedy=False, stop_strings=None, max_new_tokens=1
        )
        (Path(os.environ["SC_LP_DIAG_DIR"]) / "sampling-profile.json").write_text(
            json.dumps(
                {
                    name: getattr(sampling, name)
                    for name in (
                        "temperature",
                        "top_p",
                        "top_k",
                        "min_p",
                        "min_tokens",
                        "ignore_eos",
                        "repetition_penalty",
                        "presence_penalty",
                        "frequency_penalty",
                    )
                },
                indent=2,
            )
            + "\n"
        )

    async def reset_prefix_cache_async(self):
        return await super().reset_prefix_cache_async()

    async def diagnostic_extension_status(self):
        return await self.llm.collective_rpc("diagnostic_probe_status")

    def diagnostic_probe_status(self):
        import torch
        from nemo_rl.models.generation.vllm import vllm_worker

        kwargs = {}
        assert super().reset_prefix_cache_async.__name__ == "score_on_reset"
        vllm_worker._validate_worker_extension_cls(kwargs, "unused")
        assert kwargs["worker_extension_cls"].endswith("ProbeExtension")
        assert (
            VllmAsyncGenerationWorkerImpl.reset_prefix_cache_async.__name__
            == "score_on_reset"
        )
        assert torch.tensor([1.0, 2.0], device="cuda").square().sum().item() == 5.0
        return {
            "role": "generation",
            "interpreter": sys.executable,
            "probe_methods_installed": True,
            "device_count": torch.cuda.device_count(),
            "cuda_operation": "passed",
            "engine_extension": kwargs["worker_extension_cls"],
        }
