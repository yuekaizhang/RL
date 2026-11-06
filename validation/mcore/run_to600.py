"""Resume complete MCore state at 300; preserve the original checkpoint roots."""

import json
import os
import runpy
import sys
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
lag = int(os.environ["SC_MCORE_LOOKAHEAD"])
assert lag in (0, 1)
register_omegaconf_resolvers()
path = f"validation/mcore/la{lag}_to600.yaml"
config = MasterConfig(**OmegaConf.to_container(load_config(path), resolve=True))
validate_single_controller_config(config)
assert config.grpo.max_num_steps == 600
assert config.async_rl.sampler.max_lookahead_versions == lag
assert config.logger.wandb.id == f"yuekaiz4963mc{lag}s42"
assert config.logger.wandb.entity == "yuekaiz"
assert config.logger.wandb.resume == "must"
scheduler = config.policy["megatron_cfg"]["scheduler"]
assert scheduler["use_checkpoint_opt_param_scheduler"] is True
assert scheduler["override_opt_param_scheduler"] is False
assert scheduler["lr_decay_style"] == "constant"
assert scheduler["start_weight_decay"] == scheduler["end_weight_decay"] == 0.1
assert config.policy["logprob_chunk_size"] == 512
assert config.policy["megatron_cfg"]["checkpoint"][
    "stage_precision_aware_optimizer_state_on_cpu"
]

destination = Path(config.checkpointing["checkpoint_dir"]).resolve()
source = Path(
    f"validation/mcore/results/yuekaiz-mcore-la{lag}-300/checkpoints/step_300"
).resolve()
original_latest = CheckpointManager.get_latest_checkpoint_path


def latest_with_initial_source(self):
    """Only the first allocation falls back to the preserved source step300."""
    latest = original_latest(self)
    if latest is None and Path(self.checkpoint_dir).resolve() == destination:
        return str(source)
    return latest


CheckpointManager.get_latest_checkpoint_path = latest_with_initial_source
manager = CheckpointManager(config.checkpointing)
latest = Path(manager.get_latest_checkpoint_path())
state = json.loads((latest / "training_info.json").read_text())
step = state["current_step"]
assert state["trainer_version"] == step
if step >= 600:
    print(f"MCore LA{lag} already completed {step}; skipping", flush=True)
    sys.exit(0)
assert 300 <= step < 600, state
weights_path, optimizer_path = manager.get_resume_paths(latest)
assert weights_path is not None and optimizer_path is not None
for artifact in ("data_plane", "replay_buffer_metadata.pt", "train_dataloader.pt"):
    assert (latest / artifact).exists(), (latest, artifact)
manifest = {
    "preflight_only": "--preflight" in sys.argv,
    "lookahead": lag,
    "resume_step": step,
    "checkpoint": str(latest),
    "destination": str(destination),
    "target_steps": 600,
    "slurm_job": os.environ["SLURM_JOB_ID"],
    "wandb_id": config.logger.wandb.id,
    "config": path,
    "logprob_chunk_size": 512,
    "restore_saved_scheduler": True,
}
output = Path("validation/mcore/results")
filename = (
    f"entry-preflight-{os.environ['SLURM_JOB_ID']}-la{lag}.json"
    if "--preflight" in sys.argv
    else f"launch-{os.environ['SLURM_JOB_ID']}.json"
)
(output / filename).write_text(json.dumps(manifest, indent=2) + "\n")
print("MCORE_CONTINUE_TO600", json.dumps(manifest), flush=True)
launcher_path = (
    Path(__file__).resolve().parents[2] / "examples/run_grpo_single_controller.py"
)
if "--preflight" in sys.argv:
    # Megatron-LM has a regular `examples` package. Load our exact source file
    # without allowing it to win namespace-package resolution.
    loaded = runpy.run_path(str(launcher_path), run_name="_sc_resume_preflight")
    assert Path(loaded["__file__"]).resolve() == launcher_path.resolve()
    assert callable(loaded["main"])
    print("MCORE_RESUME_ENTRY_PREFLIGHT_PASSED", str(launcher_path), flush=True)
    sys.exit(0)
# Stay in this driver so fallback discovery remains active during setup.
sys.argv = [str(launcher_path), "--config", path]
runpy.run_path(str(launcher_path), run_name="__main__")
