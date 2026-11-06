"""Observe the submitted MCore pair without running compute on the login node."""

import argparse
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
parser = argparse.ArgumentParser()
parser.add_argument("--target-steps", type=int, choices=(300, 600), default=300)
parser.add_argument("--once", action="store_true")
args = parser.parse_args()
target = args.target_steps
suffix = "" if target == 300 else "-to600"
jobs = {str(lag): [] for lag in (0, 1)}
for line in (root / f"slurm/submitted{suffix}.tsv").read_text().splitlines():
    lag, _, job = line.split("\t")
    jobs[lag].append(job)
history_jobs = {str(lag): [] for lag in (0, 1)}
for line in (root / "slurm/submitted.tsv").read_text().splitlines():
    lag, _, job = line.split("\t")
    history_jobs[lag].append(job)
last_plot = {"0": 0, "1": 0}
last_periodic = time.monotonic()
last_progress = {"0": (300, None), "1": (300, None)}
seen_failures = set()


def notify_conversation(reason):
    """Wake the original user-authorized conversation when monitoring ends."""
    try:
        notice = subprocess.run(
            (
                ["bash", "validation/mcore/watch_10min.sh"]
                if target == 300
                else ["bash", "validation/mcore/watch_to600.sh", reason]
            ),
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=55,
        )
        print(
            "CONVERSATION_NOTIFICATION",
            reason,
            notice.returncode,
            notice.stdout.strip(),
            notice.stderr.strip(),
            flush=True,
        )
    except Exception as error:
        print("CONVERSATION_NOTIFICATION_FAILED", reason, str(error), flush=True)


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
            "target_steps": target,
            "jobs": states,
            "cases": {},
            "bad": [],
            "alerts": [],
        }
        complete = []
        for lag in jobs:
            rows = {}
            for job in history_jobs[lag]:
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
            for row in recent:
                if row["policy_kl_error"] > 0.01:
                    result["alerts"].append(
                        {
                            "lookahead": lag,
                            "step": current,
                            "reason": "policy_kl_spike",
                            "event_step": int(row["nemo_rl/step"]),
                            "policy_kl_error": row["policy_kl_error"],
                            "gen_kl_error": row["gen_kl_error"],
                        }
                    )
            if recent and (
                stats["gen_kl_error"]["mean"] > 0.01
                or stats["is_oob_ratio"]["mean"] > 0.01
            ):
                result["alerts"].append(
                    {"lookahead": lag, "step": current, "recent20": stats}
                )
            checkpoint = root / f"results/yuekaiz-mcore-la{lag}-{target}/checkpoints"
            saved = [
                json.loads(path.read_text()).get("current_step", 0)
                for path in checkpoint.glob("step_*/training_info.json")
            ]
            if target == 600 and not saved:
                saved = [300]
            done = (
                max(saved, default=0) >= target
                and current == target
                and not (set(range(1, target + 1)) - rows.keys())
            )
            complete.append(done)
            result["cases"][lag] = {
                "last_step": current,
                "last_checkpoint": max(saved, default=0),
                "complete": done,
                "recent20": stats,
                "missing_steps": sorted(set(range(1, current + 1)) - rows.keys()),
            }
            if target == 600:
                running = any(
                    states.get(job, {}).get("state") == "RUNNING" for job in jobs[lag]
                )
                previous_step, progress_time = last_progress[lag]
                if not running:
                    progress_time = None
                elif current != previous_step or progress_time is None:
                    progress_time = time.monotonic()
                elif time.monotonic() - progress_time > 1200:
                    result["alerts"].append(
                        {
                            "lookahead": lag,
                            "step": current,
                            "reason": "no_progress_20min",
                        }
                    )
                last_progress[lag] = (current, progress_time)
                if done and not args.once:
                    unused = [
                        job
                        for job in jobs[lag]
                        if states.get(job, {}).get("state") == "PENDING"
                    ]
                    if unused:
                        subprocess.run(["scancel", *unused], check=True)
        if result["bad"] and not args.once:
            subprocess.run(
                [
                    "scancel",
                    *[job for ids in jobs.values() for job in ids if job in states],
                ],
                check=True,
            )
            result["stopped_for_invalid_training"] = True
        (root / f"results/live-status{suffix}.json").write_text(
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
        if args.once:
            break
        if target == 600:
            accounting = subprocess.run(
                [
                    "sacct",
                    "-j",
                    ",".join(job for ids in jobs.values() for job in ids),
                    "-X",
                    "-n",
                    "-P",
                    "--format=JobID,State,ExitCode",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            failures = {
                line
                for line in accounting.stdout.splitlines()
                if any(
                    marker in line
                    for marker in (
                        "|FAILED|",
                        "|OUT_OF_MEMORY|",
                        "|TIMEOUT|",
                        "|NODE_FAIL|",
                    )
                )
            }
            failures.update(
                f"{job}|DependencyNeverSatisfied"
                for job, state in states.items()
                if "DependencyNeverSatisfied" in state["reason"]
            )
            failures.update(
                f"alert|{alert['lookahead']}|{alert.get('reason', 'kl_or_importance')}"
                f"|{alert.get('event_step', '')}"
                for alert in result["alerts"]
            )
            if failures - seen_failures:
                notify_conversation("failure_or_alert")
            seen_failures.update(failures)
            if time.monotonic() - last_periodic >= 600:
                notify_conversation("periodic_10min")
                last_periodic = time.monotonic()
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
                        COMMAND=f"source validation/mcore/environment.sh && uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/mcore/plot.py --target-steps {target}",
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
            if all(complete) and target == 600:
                if any(state["state"] == "RUNNING" for state in states.values()):
                    time.sleep(60)
                    continue
                (root / "results/stop-to600-watch").write_text(
                    "Both MCore cases completed600; verify final results.\n"
                )
            notify_conversation("completed" if all(complete) else "invalid_metrics")
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
            (root / f"results/final-accounting{suffix}.txt").write_text(
                accounting.stdout
            )
            print("ALLOCATIONS_ENDED_BEFORE_TARGET", accounting.stdout, flush=True)
            notify_conversation("allocations_ended_before_target")
            break
    except Exception as error:
        print(type(error).__name__, str(error), flush=True)
    time.sleep(60)
