"""Audit job's worker hook; the normal launcher never installs this hook."""

import os
import sys
from pathlib import Path


def setup():
    from nemo_rl.distributed.ray_worker_setup import ray_worker_process_setup

    ray_worker_process_setup()
    with (Path(os.environ["SC_AUDIT_DIR"]) / "worker-hooks.log").open("a") as stream:
        stream.write(f"{os.getpid()} {sys.prefix} {sys.executable}\n")
    from nemo_rl.models.generation.vllm import vllm_worker

    original = vllm_worker._validate_worker_extension_cls

    def inject_extension(kwargs, required):
        original(kwargs, required)
        kwargs["worker_extension_cls"] = (
            "validation.correctness.vllm_probe.AuditWorkerExtension"
        )

    vllm_worker._validate_worker_extension_cls = inject_extension
