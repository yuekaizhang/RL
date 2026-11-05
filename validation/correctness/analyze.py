"""Assert group ownership, token lineage, actual TIS inputs and overlap evidence."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def rows(path):
    payload = torch.load(path, weights_only=False, map_location="cpu")
    data = payload["data"]
    return {
        sid: {
            key: value.unbind()[i].detach().cpu()
            for key, value in data.items()
            if isinstance(value, torch.Tensor)
        }
        for i, sid in enumerate(payload["ids"])
    }


def trimmed(row, key):
    length = int(row["input_lengths"].item())
    return row[key][:length]


def analyze(root):
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    # Work finishing after the saved cut in the stopped process was not durable.
    # It is legal to redo it. Compare the durable prefix with the resumed run.
    init = [e for e in events if e["event"] == "init"]
    assert len(init) >= 2 and init[1]["restored_step"] > 0
    boundaries = {}
    for before, after in zip(init, init[1:]):
        boundaries[before["session"]] = next(
            e["time_ns"]
            for e in events
            if e["event"] == "checkpoint_cut"
            and e["session"] == before["session"]
            and e["step"] == after["restored_step"]
        )
    events = [
        e
        for e in events
        if e["session"] not in boundaries or e["time_ns"] <= boundaries[e["session"]]
    ]
    dispatches = [e for e in events if e["event"] == "dispatch"]
    targets = {e["group"]: e["target"] for e in dispatches}
    assert all(0 <= e["target"] - e["version"] <= 1 for e in dispatches)
    assert max(e["buffered"] for e in dispatches) <= 16
    assert any(e["target"] > e["version"] for e in dispatches)
    admitted = [e for e in events if e["event"] == "admit"]
    assert len({e["group"] for e in admitted}) == len(admitted)
    prompt_keys = [(e["target"], e["prompt"]) for e in admitted]
    assert len(set(prompt_keys)) == len(prompt_keys)
    selected = [e for e in events if e["event"] == "select"]
    consumed = [sid for e in events if e["event"] == "consume" for sid in e["ids"]]
    selection = [sid for e in selected for sid in e["ids"]]
    assert len(set(consumed)) == len(consumed)
    assert Counter(selection) == Counter(consumed)
    assert [e["step"] for e in selected] == list(range(len(selected)))
    admission_by_group = {e["group"]: e for e in admitted}
    used_groups = {tag["group_id"] for e in selected for tag in e["tags"]}
    due_groups = {e["group"] for e in admitted if e["target"] < len(selected)}
    assert due_groups == used_groups, (
        "An admitted prompt group was skipped or an unadmitted group was trained"
    )
    for e in selected:
        groups = {tag["group_id"] for tag in e["tags"]}
        assert len(groups) == 8 and len(e["ids"]) == 32
        assert all(targets[group] == e["version"] for group in groups)
        assert all(0 <= e["version"] - tag["weight_version"] <= 1 for tag in e["tags"])
        assert all(
            str(tag["prompt_idx"]) == admission_by_group[tag["group_id"]]["prompt"]
            for tag in e["tags"]
        )
    versioned_ready = [
        e for e in events if e["event"] == "ready" and e.get("start_weight") is not None
    ]
    assert versioned_ready
    for e in versioned_ready:
        assert 0 <= e["end_weight"] - e["start_weight"] <= 1
        assert e["end_weight"] <= e["version"]
        assert all(tag["weight_version"] == e["start_weight"] for tag in e["tags"])

    active = {}
    intervals = []
    training = []
    open_train = {}
    for e in events:
        if e["event"] == "rpc_start":
            active[e["request"]] = e
        elif e["event"] in ("rpc_output", "rpc_end"):
            start = active.pop(e["request"], None)
            if start is not None:
                intervals.append((start["time_ns"], e["time_ns"], start["group"]))
        elif e["event"] == "train_start":
            open_train[(e["session"], e["step"])] = e
        elif e["event"] == "train_end":
            start = open_train.pop((e["session"], e["step"]))
            training.append((start["time_ns"], e["time_ns"], e["step"]))
    overlap = [
        {
            "step": step,
            "future_group": group,
            "overlap_ms": (min(b, d) - max(a, c)) / 1e6,
        }
        for a, b, step in training
        for c, d, group in intervals
        if targets[group] > step and min(b, d) > max(a, c)
    ]
    assert overlap, "No observed overlap of policy F/B with a future generation RPC"
    gpu_steps = [
        json.loads(line)
        for line in (root / "generation-gpu.jsonl").read_text().splitlines()
    ]
    gpu_overlap = [
        {
            "step": step,
            "cuda_ms": gpu["cuda_ms"],
            "scheduled_tokens": gpu["tokens"],
            "start_ns": gpu["start_ns"],
            "end_ns": gpu["end_ns"],
        }
        for a, b, step in training
        for gpu in gpu_steps
        if gpu["tokens"] > 0
        and gpu["cuda_ms"] > 0
        and a <= gpu["start_ns"] < gpu["end_ns"] <= b
        and any(
            c <= gpu["start_ns"] < gpu["end_ns"] <= d and targets[group] > step
            for c, d, group in intervals
        )
    ]
    assert gpu_overlap, (
        "No synchronized generation GPU execution fully contained in policy F/B"
    )
    cuts = [
        e
        for e in events
        if e["event"] == "checkpoint_cut" and e["ready"] and e["pending_rpcs"]
    ]
    assert cuts, (
        "No exclusive checkpoint cut contains BOTH canonical ready rows and pending generation RPCs"
    )
    restore_step = init[1]["restored_step"]
    cut = next(e for e in cuts if e["step"] == restore_step)
    following_sessions = {e["session"] for e in init[1:]}
    after = {
        tag["group_id"]
        for e in selected
        if e["session"] in following_sessions
        for tag in e["tags"]
    }
    owned = set(cut["ready"]) | set(cut["pending_rpcs"].values())
    # End-of-run lookahead work remains unconsumed deliberately. Assert ownership
    # only for target steps actually reached by this bounded test.
    expected = {g for g in owned if targets[g] < len(selected)}
    assert expected <= after

    published = {}
    for path in root.glob("published-*.pt"):
        published.update(rows(path))
    selected_rows = {}
    for path in root.glob("selected-*.pt"):
        for sid, row in rows(path).items():
            source = published[sid]
            for key in ("input_ids", "generation_logprobs", "token_mask"):
                torch.testing.assert_close(
                    trimmed(row, key), trimmed(source, key), rtol=0, atol=0
                )
            selected_rows[sid] = row
    assert len(selected_rows) == len(selection)
    by_tokens = {}
    for sid, row in selected_rows.items():
        key = tuple(trimmed(row, "input_ids").tolist())
        by_tokens.setdefault(key, []).append((sid, row))
    matched = Counter()
    loss_calls = 0
    max_forward_gap = 0.0
    for path in (root / "loss").glob("*.pt"):
        call = torch.load(path, weights_only=False, map_location="cpu")
        data = call["data"]
        effective_generation = torch.where(
            call["mask"].bool(), data["generation_logprobs"][:, 1:], 0.0
        )
        importance = (
            (call["prev"] - effective_generation).exp().nan_to_num(0, 0, 0).clamp(0, 2)
        )
        # The in-worker CUDA comparison is exact. CPU exp can differ by one
        # float32 ULP when independently reconstructing the saved tensors.
        torch.testing.assert_close(call["importance"], importance, rtol=3e-7, atol=1e-7)
        for i in range(data["input_ids"].shape[0]):
            length = int(data["input_lengths"][i])
            tokens = tuple(data["input_ids"][i, :length].tolist())
            candidates = by_tokens[tokens]
            compatible = []
            for sid, source in candidates:
                if torch.equal(
                    data["prev_logprobs"][i, 1:length],
                    source["prev_logprobs"][1:length],
                ):
                    compatible.append((sid, source))
            assert compatible, (
                "Worker loss received policy LPs from a different row/version"
            )
            # Identical sampled sequences can occur. Resolve against remaining
            # multiplicity and require exact generation LP/mask agreement too.
            sid, source = next(
                (sid, row) for sid, row in compatible if matched[sid] == 0
            )
            torch.testing.assert_close(
                data["generation_logprobs"][i, 1:length],
                source["generation_logprobs"][1:length],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                data["token_mask"][i, :length],
                source["token_mask"][:length],
                rtol=0,
                atol=0,
            )
            matched[sid] += 1
        mask = call["mask"].bool()
        torch.testing.assert_close(
            call["mask"],
            data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1),
            rtol=0,
            atol=0,
        )
        if mask.any():
            max_forward_gap = max(
                max_forward_gap,
                float((call["current"] - call["prev"])[mask].abs().max()),
            )
        loss_calls += 1
    assert sum(matched.values()) == len(selection)
    assert all(count == 1 for count in matched.values())
    return {
        "status": "passed",
        "steps": len(selected),
        "samples_consumed_once": len(consumed),
        "admitted_groups_due_and_consumed_once": len(used_groups),
        "ready_groups_with_verified_weight_interval": len(versioned_ready),
        "groups_admitted": len(admitted),
        "max_buffered_groups": max(e["buffered"] for e in dispatches),
        "future_rpc_train_overlap_count": len(overlap),
        "overlap_examples": overlap[:5],
        "generation_gpu_executions_inside_training": len(gpu_overlap),
        "generation_gpu_overlap_examples": gpu_overlap[:5],
        "restore_step": restore_step,
        "ready_groups_at_cut": cut["ready"],
        "pending_rpc_groups_at_cut": sorted(set(cut["pending_rpcs"].values())),
        "restored_groups_consumed": sorted(expected),
        "loss_microbatches": loss_calls,
        "max_current_vs_recomputed_policy_logprob_gap": max_forward_gap,
        "resident_weight_check": "separate resident-report.json",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
