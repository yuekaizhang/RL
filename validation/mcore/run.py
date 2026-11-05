"""Start or resume one of the two 300-step MCore comparisons."""

import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.utils.checkpoint import CheckpointManager

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
lag = int(os.environ["SC_MCORE_LOOKAHEAD"])
assert lag in (0, 1)
register_omegaconf_resolvers()
path = f"validation/mcore/la{lag}.yaml"
config = OmegaConf.to_container(load_config(path), resolve=True)
validate_single_controller_config(MasterConfig(**config))
checkpoint = Path(config["checkpointing"]["checkpoint_dir"])
saved = sorted(
    checkpoint.glob("step_*/training_info.json"),
    key=lambda p: int(p.parent.name.split("_")[-1]),
)
step = 0
if saved:
    latest = saved[-1]
    state = json.loads(latest.read_text())
    step = state["current_step"]
    assert state["trainer_version"] == step
    # MCore embeds optimizer state in the distributed weights checkpoint.
    # The returned optimizer path is a loader flag, not a separate directory.
    weights_path, optimizer_path = CheckpointManager.get_resume_paths(latest.parent)
    assert weights_path is not None and optimizer_path is not None
    assert (latest.parent / "data_plane").exists()
if step >= 300:
    print(f"MCore LA{lag} already completed {step}; skipping")
    sys.exit(0)
manifest = {
    "lookahead": lag,
    "resume_step": step,
    "target_steps": 300,
    "slurm_job": os.environ["SLURM_JOB_ID"],
    "wandb_id": config["logger"]["wandb"]["id"],
    "config": path,
    "logprob_chunk_size": config["policy"]["logprob_chunk_size"],
}
output = Path("validation/mcore/results")
output.mkdir(exist_ok=True, parents=True)
(output / f"launch-{os.environ['SLURM_JOB_ID']}.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print(json.dumps(manifest), flush=True)
command = [sys.executable, "examples/run_grpo_single_controller.py", "--config", path]
if step:
    command.append("logger.wandb.resume=must")
os.execv(sys.executable, command)
