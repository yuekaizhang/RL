"""Isolated worker instrumentation; ordinary experiment launchers do not load it."""

import copy
import json
import os
from pathlib import Path


def setup():
    from nemo_rl.distributed.ray_worker_setup import ray_worker_process_setup

    ray_worker_process_setup()


def install_policy():
    """Install after Ray assigns actor GPU visibility, never in process setup."""
    import torch

    from nemo_rl.models.policy.workers import automodel_policy_worker as module
    from validation.kl_diagnosis.sampling import sample_tensor

    cls = module.AutomodelPolicyWorkerImpl
    if cls.get_logprobs.__name__ == "record_lp":
        return
    get_lp = cls.get_logprobs
    finish = cls.finish_train_step
    refit = cls._refit_params_generator
    refit_dtype = module._refit_tensor_dtype
    ssm_values = {}

    def record_dtype(name, tensor, default_dtype):
        dtype = refit_dtype(name, tensor, default_dtype)
        # Reproduce the old wire precision only in this isolated A/B experiment.
        # Ordinary workers keep the production FP32 A_log preservation fix.
        if (
            os.environ.get("SC_LP_FORCE_LEGACY_A_LOG") == "1"
            and name.endswith(".A_log")
            and tensor.dtype == torch.float32
        ):
            dtype = default_dtype
        if name.endswith((".A_log", ".dt_bias")) and torch.distributed.get_rank() == 0:
            ssm_values[name] = tensor.detach().cpu().clone()
            error = (tensor.float() - tensor.to(dtype).float()).abs().max().item()
            path = Path(os.environ["SC_LP_DIAG_DIR"]) / "ssm-wire-dtype.jsonl"
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "name": name,
                            "source": str(tensor.dtype),
                            "wire": str(dtype),
                            "max_abs_error": error,
                        }
                    )
                    + "\n"
                )
        return dtype

    def record_lp(self, data, *args, **kwargs):
        result = get_lp(self, data, *args, **kwargs)
        if not getattr(self, "_diag_replaying", False):
            self._diag_data = copy.deepcopy(data)
            self._diag_data.to("cpu")
            self._diag_before = result["logprobs"].clone()
        return result

    def finish_and_replay(self):
        result = finish(self)
        self._diag_replaying = True
        try:
            self.prepare_for_lp_inference()
            after = get_lp(self, self._diag_data)["logprobs"]
            root = Path(os.environ["SC_LP_DIAG_DIR"]) / "drift"
            root.mkdir(exist_ok=True)
            count = getattr(self, "_diag_update", 0)
            self._diag_update = count + 1
            torch.save(
                {"data": self._diag_data, "before": self._diag_before, "after": after},
                root / f"{count}-{torch.distributed.get_rank()}.pt",
            )
        finally:
            self._diag_replaying = False
        return result

    def record_refit(self):
        rank = torch.distributed.get_rank()
        count = getattr(self, "_diag_refit", 0)
        self._diag_refit = count + 1
        values = {}
        for name, tensor in refit(self):
            if rank == 0:
                values[name] = sample_tensor(tensor)
            yield name, tensor
        if rank == 0:
            torch.save(
                values, Path(os.environ["SC_LP_DIAG_DIR"]) / f"sender-{count}.pt"
            )
            torch.save(
                ssm_values,
                Path(os.environ["SC_LP_DIAG_DIR"]) / f"ssm-source-{count}.pt",
            )

    cls.get_logprobs = record_lp
    cls.finish_train_step = finish_and_replay
    cls._refit_params_generator = record_refit
    module._refit_tensor_dtype = record_dtype


def install_generation():
    """Patch module globals in the actual actor process, after deserialization."""
    from nemo_rl.models.generation.vllm import vllm_worker
    from validation.kl_diagnosis.scoring import install

    install()
    original = vllm_worker._validate_worker_extension_cls
    if original.__name__ == "inject_extension":
        return

    def inject_extension(kwargs, required):
        original(kwargs, required)
        kwargs["worker_extension_cls"] = (
            "validation.kl_diagnosis.weights.ProbeExtension"
        )

    vllm_worker._validate_worker_extension_cls = inject_extension
