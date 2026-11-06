"""Read the preserved checkpoint; write only to the isolated diagnostic root."""

import os
from pathlib import Path

from examples import run_grpo_single_controller as launcher
from nemo_rl.distributed import virtual_cluster
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.utils.checkpoint import CheckpointManager
from validation.kl_diagnosis.runtime import DiagnosticController

source = Path(
    "validation/results/yuekaiz-qwen35-35ba3b-ep16-fullbatch-la1-s42-pr4963"
    "/checkpoints/step_600"
).resolve()
assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
assert (source / "policy/optimizer").is_dir()
# Patch discovery in this driver process only. No checkpoint symlinks or writes
# under the preserved convergence run are required. The actor gets the explicit
# last_checkpoint_path through the ordinary setup arguments.
CheckpointManager.get_latest_checkpoint_path = lambda self: str(source)
virtual_cluster._WORKER_PROCESS_SETUP_HOOK = (
    "validation.kl_diagnosis.worker_setup.setup"
)
launcher.SingleControllerActor = DiagnosticController
ACTOR_ENVIRONMENT_REGISTRY[
    "validation.kl_diagnosis.policy_worker.DiagnosticPolicyWorker"
] = "/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python"
ACTOR_ENVIRONMENT_REGISTRY[
    "validation.kl_diagnosis.generation_worker.DiagnosticGenerationWorker"
] = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"
for fqn in (
    "validation.kl_diagnosis.policy_worker.DiagnosticPolicyWorker",
    "validation.kl_diagnosis.generation_worker.DiagnosticGenerationWorker",
):
    executable = ACTOR_ENVIRONMENT_REGISTRY[fqn]
    assert Path(executable).is_file(), (fqn, executable)
    assert not executable.startswith("uv"), (fqn, executable)
    print("DIAGNOSTIC_ACTOR_INTERPRETER", fqn, executable, flush=True)
launcher.main()
