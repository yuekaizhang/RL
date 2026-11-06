"""Compare MCore LA0/LA1 and historical AutoModel curves."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from validation.scripts.read_metrics import read_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--target-steps", type=int, choices=(300, 600), default=300)
target = parser.parse_args().target_steps
repo = Path(__file__).resolve().parents[2]
output = repo / "validation/mcore/results"
output.mkdir(parents=True, exist_ok=True)
submitted = repo / "validation/mcore/slurm/submitted.tsv"
jobs = {str(lag): [] for lag in (0, 1)}
if submitted.exists():
    for line in submitted.read_text().splitlines():
        lag, _, job = line.split("\t")
        jobs[lag].append(job)
series = {}
for lag in (0, 1):
    records = {}
    for job in jobs[str(lag)]:
        path = repo / f"validation/mcore/slurm/{job}-logs/ray-driver.log"
        if path.exists():
            for row in read_metrics(path):
                if "nemo_rl/step" in row:
                    records[int(row["nemo_rl/step"])] = {
                        "nemo_rl/step": row["nemo_rl/step"],
                        **{
                            (
                                key if key.startswith("timing/") else f"train/{key}"
                            ): value
                            for key, value in row.items()
                            if key != "nemo_rl/step"
                        },
                    }
    series[f"MCore LA{lag}"] = list(records.values())
    history_name = f"la{lag}-history" + ("-to600" if target == 600 else "")
    (output / f"{history_name}.json").write_text(
        json.dumps(series[f"MCore LA{lag}"], indent=2) + "\n"
    )
for label, path in [
    (
        "AutoModel LA0 (historical)",
        repo / "session/20261102_233142/c926b37d-history.json",
    ),
    ("AutoModel LA1 (historical)", repo / "validation/results/la1-history.json"),
]:
    if path.exists():
        series[label] = json.loads(path.read_text())
metrics = [
    "train/gen_kl_error",
    "train/reward",
    "train/mean_gen_tokens_per_sample",
    "train/loss",
    "train/approx_entropy",
    "train/grad_norm",
    "validation/acc",
    "timing/train/total_step_time",
]
figure, axes = plt.subplots(4, 2, figsize=(15, 16))
summary = {}
for label, rows in series.items():
    steps = sorted(
        {
            int(row["nemo_rl/step"])
            for row in rows
            if isinstance(row.get("nemo_rl/step"), (int, float))
            and 1 <= row["nemo_rl/step"] <= target
        }
    )
    summary[label] = {
        "last_step": max(steps, default=0),
        "missing_steps": sorted(set(range(1, max(steps, default=0) + 1)) - set(steps)),
        "metrics": {},
    }
colors = ["#2874a6", "#e67e22", "#27ae60", "#a93226"]
for ax, metric in zip(axes.flat, metrics):
    plotted = False
    for color, (label, rows) in zip(colors, series.items()):
        values = dict(
            sorted(
                {
                    int(row["nemo_rl/step"]): float(row[metric])
                    for row in rows
                    if isinstance(row.get("nemo_rl/step"), (int, float))
                    and 1 <= row["nemo_rl/step"] <= target
                    and isinstance(row.get(metric), (int, float))
                }.items()
            )
        )
        if not values:
            continue
        plotted = True
        steps, ys = list(values), list(values.values())
        averages = [
            statistics.mean(ys[max(0, index - 19) : index + 1])
            for index in range(len(ys))
        ]
        ax.plot(steps, ys, color=color, alpha=0.18, linewidth=0.6)
        ax.plot(
            steps,
            averages,
            color=color,
            linestyle="--" if "historical" in label else "-",
            label=label,
            linewidth=1.6,
        )
        summary[label]["metrics"][metric] = {
            "n": len(ys),
            "mean": statistics.mean(ys),
            "last20_mean": statistics.mean(ys[-20:]),
            "last100_mean": statistics.mean(ys[-100:]),
            "max": max(ys),
        }
    ax.set(title=metric, xlabel="Optimizer step", xlim=(1, target))
    ax.grid(alpha=0.2)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "Not recorded: online validation disabled",
            transform=ax.transAxes,
            ha="center",
        )
figure.suptitle(
    "Qwen3.5-35B-A3B full-batch SC: MCore LA0 / LA1\n20-step means; AutoModel curves are historical, single-seed references"
)
figure.tight_layout(rect=(0, 0, 1, 0.96))
name = "comparison" if target == 300 else "comparison-to600"
figure.savefig(output / f"{name}.png", dpi=140)
plt.close(figure)
(output / f"{name}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    json.dumps({name: {"last_step": row["last_step"]} for name, row in summary.items()})
)
