"""Compare actual vLLM post-refit parameters with saved AutoModel weights."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def check(root, step):
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    start = [e for e in events if e["event"] == "refit_start" and e["version"] == step][
        -1
    ]
    end = next(
        e
        for e in events
        if e["event"] == "refit_end"
        and e["session"] == start["session"]
        and e["time_ns"] > start["time_ns"]
    )
    snapshots = [
        torch.load(p, weights_only=True) for p in (root / "resident").glob("*.pt")
    ]
    candidates = [
        s for s in snapshots if start["time_ns"] <= s["time_ns"] <= end["time_ns"]
    ]
    assert len(candidates) == 1, (
        f"Expected one TP1 resident snapshot inside version {step} refit, got {len(candidates)}"
    )
    resident = candidates[0]["weights"]
    previous = max(
        (s for s in snapshots if s["time_ns"] < start["time_ns"]),
        key=lambda s: s["time_ns"],
    )
    changed = sum(
        int((value != previous["weights"][name]).sum())
        for name, value in resident.items()
    )
    assert changed > 0, (
        "The probed weights did not distinguish this version from the previous refit"
    )
    checkpoint = root / "ckpts" / f"step_{step}" / "policy/weights/model"
    matched = set()
    casts = {}
    for path in checkpoint.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name not in resident:
                    continue
                view = handle.get_slice(name)
                shape = view.get_shape()
                value = view[0, :256] if len(shape) == 2 else view[:256]
                if value.dtype != resident[name].dtype:
                    casts[name] = [str(value.dtype), str(resident[name].dtype)]
                torch.testing.assert_close(
                    value.reshape(-1).to(resident[name].dtype),
                    resident[name],
                    rtol=0,
                    atol=0,
                )
                matched.add(name)
    assert matched == set(resident), set(resident) - matched
    assert len(matched) >= 50
    result = {
        "status": "passed",
        "version": step,
        "parameter_slices_compared_exactly": len(matched),
        "values_compared": sum(t.numel() for t in resident.values()),
        "values_changed_since_previous_refit": changed,
        "parameters": sorted(matched),
        "precision_conversions": casts,
        "refit_interval_ns": [start["time_ns"], end["time_ns"]],
    }
    (root / "resident-report.json").write_text(json.dumps(result, indent=2) + "\n")
    # Retain the small expected slices so the result remains auditable after
    # temporary correctness checkpoints are removed.
    torch.save(
        {"version": step, "weights": resident}, root / "verified-resident-weights.pt"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "parameters"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("step", type=int)
    args = parser.parse_args()
    check(args.root, args.step)
