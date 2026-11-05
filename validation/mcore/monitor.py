"""Observe the submitted MCore pair without running compute on the login node."""

import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

from validation.scripts.read_metrics import read_metrics

repo = Path(__file__).resolve().parents[2]
root = repo / "validation/mcore"
jobs = {str(lag): [] for lag in (0, 1)}
for line in (root / "slurm/submitted.tsv").read_text().splitlines():
    lag, _, job = line.split("\t")
    jobs[lag].append(job)
last_plot = {"0": 0, "1": 0}
while True:
    try:
        queue = subprocess.run(
            [
                "squeue",
                "-h",
                "-j",
                ",".join(job for ids in jobs.values() for job in ids),
                "-o",
                "%i|%T|%R",
            ],
            capture_output=True,
            text=True,
        )
        if queue.returncode:
            raise RuntimeError(queue.stderr.strip())
        states = {
            parts[0]: {"state": parts[1], "reason": parts[2]}
            for line in queue.stdout.splitlines()
            if len(parts := line.split("|", 2)) == 3
        }
        result = {
            "updated_epoch": time.time(),
            "jobs": states,
            "cases": {},
            "bad": [],
            "alerts": [],
        }
        complete = []
        for lag, ids in jobs.items():
            rows = {}
            for job in ids:
                path = root / f"slurm/{job}-logs/ray-driver.log"
                if path.exists():
                    for row in read_metrics(path):
                        if "nemo_rl/step" in row:
                            rows[int(row["nemo_rl/step"])] = row
            current = max(rows, default=0)
            recent = [rows[step] for step in sorted(rows)[-20:]]
            for step, row in rows.items():
                for metric in [
                    "gen_kl_error",
                    "policy_kl_error",
                    "loss",
                    "grad_norm",
                    "reward",
                    "approx_entropy",
                ]:
                    if not math.isfinite(row[metric]):
                        result["bad"].append(
                            {
                                "lookahead": lag,
                                "step": step,
                                "metric": metric,
                                "value": str(row[metric]),
                            }
                        )
                if row["weight_version_lag"] not in ([0] if lag == "0" else [0, 1]):
                    result["bad"].append(
                        {
                            "lookahead": lag,
                            "step": step,
                            "metric": "weight_version_lag",
                            "value": row["weight_version_lag"],
                        }
                    )
                if (
                    row["dropped_prompt_groups"]
                    or row["num_masked_seqs_by_logprob_error"]
                    or row["num_valid_samples"] != 32
                ):
                    result["bad"].append(
                        {
                            "lookahead": lag,
                            "step": step,
                            "metric": "sample_ownership_or_mask",
                            "valid": row["num_valid_samples"],
                        }
                    )
                expected_lr = 1e-14 + (1e-6 - 1e-14) * min(step - 1, 10) / 10
                if not math.isclose(
                    row["lr"], expected_lr, rel_tol=1e-7, abs_tol=1e-16
                ):
                    result["bad"].append(
                        {
                            "lookahead": lag,
                            "step": step,
                            "metric": "lr",
                            "value": row["lr"],
                            "expected": expected_lr,
                        }
                    )
            stats = (
                {
                    metric: {
                        "mean": statistics.mean(row[metric] for row in recent),
                        "max": max(row[metric] for row in recent),
                    }
                    for metric in [
                        "gen_kl_error",
                        "policy_kl_error",
                        "reward",
                        "mean_gen_tokens_per_sample",
                        "is_oob_ratio",
                    ]
                }
                if recent
                else {}
            )
            if recent and (
                stats["gen_kl_error"]["mean"] > 0.01
                or stats["is_oob_ratio"]["mean"] > 0.01
            ):
                result["alerts"].append(
                    {"lookahead": lag, "step": current, "recent20": stats}
                )
            checkpoint = root / f"results/yuekaiz-mcore-la{lag}-300/checkpoints"
            saved = [
                json.loads(path.read_text()).get("current_step", 0)
                for path in checkpoint.glob("step_*/training_info.json")
            ]
            done = (
                max(saved, default=0) >= 300
                and current == 300
                and not (set(range(1, 301)) - rows.keys())
            )
            complete.append(done)
            result["cases"][lag] = {
                "last_step": current,
                "last_checkpoint": max(saved, default=0),
                "complete": done,
                "recent20": stats,
                "missing_steps": sorted(set(range(1, current + 1)) - rows.keys()),
            }
        if result["bad"]:
            subprocess.run(
                [
                    "scancel",
                    *[job for ids in jobs.values() for job in ids if job in states],
                ],
                check=True,
            )
            result["stopped_for_invalid_training"] = True
        (root / "results/live-status.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "steps": {
                        lag: case["last_step"] for lag, case in result["cases"].items()
                    },
                    "queue": states,
                    "bad": result["bad"],
                    "alerts": result["alerts"],
                }
            ),
            flush=True,
        )
        if any(
            result["cases"][lag]["last_step"] >= last_plot[lag] + 10 for lag in jobs
        ):
            for job, state in states.items():
                if (
                    state["state"] == "RUNNING"
                    and (root / f"slurm/{job}-logs/STARTED_RAY_HEAD").exists()
                    and (repo / f"{job}-attach.sh").exists()
                ):
                    env = dict(
                        os.environ,
                        COMMAND="source validation/mcore/environment.sh && uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/mcore/plot.py",
                    )
                    plotted = subprocess.run(
                        ["bash", f"{job}-attach.sh"],
                        cwd=repo,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if plotted.returncode == 0:
                        last_plot = {
                            lag: case["last_step"]
                            for lag, case in result["cases"].items()
                        }
                    break
        if all(complete) or result["bad"]:
            break
        if not states:
            accounting = subprocess.run(
                [
                    "sacct",
                    "-j",
                    ",".join(job for ids in jobs.values() for job in ids),
                    "-X",
                    "--noheader",
                    "--parsable2",
                    "--format=JobID,State,ExitCode",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            (root / "results/final-accounting.txt").write_text(accounting.stdout)
            print("ALLOCATIONS_ENDED_BEFORE_TARGET", accounting.stdout, flush=True)
            break
    except Exception as error:
        print(type(error).__name__, str(error), flush=True)
    time.sleep(60)
