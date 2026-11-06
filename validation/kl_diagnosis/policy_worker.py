"""Normal actor import timing with isolated policy probes."""

import json
import os
import sys
from pathlib import Path

import ray
import torch

from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
)


@ray.remote
class DiagnosticPolicyWorker(AutomodelPolicyWorkerImpl):  # pragma: no cover
    def __init__(self, *args, probe_only=False, **kwargs):
        from validation.kl_diagnosis.worker_setup import install_policy

        install_policy()
        assert "automodel_policy_worker.AutomodelPolicyWorker" in sys.prefix, sys.prefix
        binding = {
            "interpreter": sys.executable,
            "rank": os.environ["RANK"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ray_gpu_ids": ray.get_gpu_ids(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
        }
        root = Path(os.environ["SC_LP_DIAG_DIR"])
        (root / f"binding-{binding['rank']}.json").write_text(
            json.dumps(binding, indent=2) + "\n"
        )
        assert binding["device_count"] == 1, binding
        assert binding["current_device"] == 0, binding
        if probe_only:
            return
        super().__init__(*args, **kwargs)

    def get_logprobs(self, *args, **kwargs):
        return super().get_logprobs(*args, **kwargs)

    def finish_train_step(self):
        return super().finish_train_step()

    def _refit_params_generator(self):
        yield from super()._refit_params_generator()

    def diagnostic_probe_status(self):
        from nemo_rl.models.policy.workers import automodel_policy_worker as module

        assert module._refit_tensor_dtype.__name__ == "record_dtype"
        assert super().get_logprobs.__name__ == "record_lp"
        assert super()._refit_params_generator.__name__ == "record_refit"
        assert module.AutomodelPolicyWorkerImpl.get_logprobs.__name__ == "record_lp"
        assert (
            module.AutomodelPolicyWorkerImpl._refit_params_generator.__name__
            == "record_refit"
        )
        assert torch.tensor([1.0, 2.0], device="cuda").square().sum().item() == 5.0
        return {
            "role": "policy",
            "interpreter": sys.executable,
            "probe_methods_installed": True,
            "device_count": torch.cuda.device_count(),
            "cuda_operation": "passed",
            "module_dtype_probe": True,
        }
