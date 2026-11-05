"""Launch the ordinary recipe with the isolated audited controller."""

from examples import run_grpo_single_controller as launcher
from nemo_rl.distributed import virtual_cluster
from validation.correctness.audit_runtime import AuditedController

virtual_cluster._WORKER_PROCESS_SETUP_HOOK = "validation.correctness.worker_setup.setup"
launcher.SingleControllerActor = AuditedController
launcher.main()
