"""Audit-only vLLM extension: record actual resident weights after each refit."""

import os
import json
import time
from pathlib import Path

import torch

from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension


class AuditWorkerExtension(VllmInternalWorkerExtension):
    """Preserve refit behavior and save small, comparable parameter slices."""

    def audit_resident_weights(self):
        values = {}
        for name, value in self._get_named_parameters().items():
            # TP=1 in this correctness recipe. These parameters have the same
            # layout in HF/AutoModel and vLLM, unlike fused QKV/MLP matrices.
            if (
                name.endswith("embed_tokens.weight")
                or "layernorm.weight" in name
                or name == "model.norm.weight"
            ):
                values[name] = value.detach().reshape(-1)[:256].cpu().clone()
        assert values
        path = Path(os.environ["SC_AUDIT_DIR"]) / "resident"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"time_ns": time.monotonic_ns(), "weights": values},
            path / f"{time.monotonic_ns()}.pt",
        )
        if not getattr(self, "_audit_gpu_probe_installed", False):
            self._audit_gpu_probe_installed = True
            original = self.model_runner.execute_model
            output = path.parent / "generation-gpu.jsonl"

            def execute_model(*args, **kwargs):
                scheduler = kwargs.get("scheduler_output", args[0] if args else None)
                tokens = getattr(scheduler, "total_num_scheduled_tokens", 0)
                torch.cuda.synchronize()
                start = time.monotonic_ns()
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                result = original(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                finish = time.monotonic_ns()
                with output.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "start_ns": start,
                                "end_ns": finish,
                                "cuda_ms": begin.elapsed_time(end),
                                "tokens": tokens,
                            }
                        )
                        + "\n"
                    )
                return result

            self.model_runner.execute_model = execute_model

    def update_weights_from_collective(self, *args, **kwargs):
        result = VllmInternalWorkerExtension.update_weights_from_collective(
            self, *args, **kwargs
        )
        self.audit_resident_weights()
        return result

    def update_weights_via_ipc_zmq(self, *args, **kwargs):
        result = VllmInternalWorkerExtension.update_weights_via_ipc_zmq(
            self, *args, **kwargs
        )
        self.audit_resident_weights()
        return result
