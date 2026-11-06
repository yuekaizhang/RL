"""Exercise the actual image scheduler's 300-to-600 checkpoint restore path."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")


def make(horizon):
    optimizer = SimpleNamespace(param_groups=[{}])
    scheduler = OptimizerParamScheduler(
        optimizer=optimizer,
        init_lr=1e-14,
        max_lr=1e-6,
        min_lr=1e-6,
        lr_warmup_steps=10 * 32,
        lr_decay_steps=300 * 32,
        lr_decay_style="constant",
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=horizon * 32,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=True,
        override_opt_param_scheduler=False,
    )
    return optimizer, scheduler


_, original = make(300)
original.step(increment=300 * 32)
optimizer, restored = make(600)
restored.load_state_dict(original.state_dict())
assert restored.num_steps == 300 * 32
assert restored.wd_incr_steps == original.wd_incr_steps == 300 * 32
for _ in range(300):
    assert optimizer.param_groups[0]["lr"] == 1e-6
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    restored.step(increment=32)
assert restored.num_steps == 600 * 32
report = {
    "status": "passed",
    "scope": "actual MCore scheduler API with synthetic checkpoint state; full model/optimizer restore checked by training job",
    "resume_counter": 300 * 32,
    "final_counter": restored.num_steps,
    "preserved_wd_horizon": restored.wd_incr_steps,
    "lr": 1e-6,
    "weight_decay": 0.1,
    "updates_checked": 300,
}
Path(
    f"validation/mcore/results/to600-scheduler-check-{os.environ['SLURM_JOB_ID']}.json"
).write_text(json.dumps(report, indent=2) + "\n")
print("MCORE_SCHEDULER_TO600", json.dumps(report), flush=True)
