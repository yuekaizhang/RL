"""Analyze same-weight same-token request-scheduling repeatability in allocation."""

import itertools
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
root = Path(sys.argv[1])


def read(name):
    return torch.load(root / name, map_location="cpu", weights_only=False)


def stats(values):
    x = torch.cat(values).double()
    assert torch.isfinite(x).all()
    return {
        "tokens": x.numel(),
        "mean_k3": (x.exp() - 1 - x).mean().item(),
        "reverse_mean_k3": ((-x).exp() - 1 + x).mean().item(),
        "rms_logprob_delta": x.square().mean().sqrt().item(),
        "max_abs_delta": x.abs().max().item(),
        "nonidentical_fraction": (x != 0).double().mean().item(),
    }


selected = read("selected-600.pt")
assert selected["train_step"] == selected["trainer_version"] == 600
assert len(selected["ids"]) == len(set(selected["ids"])) == 32
assert {tag["weight_version"] for tag in selected["tags"]} == {600}
scoring_selection = (
    read("scoring-reference.pt")
    if (root / "scoring-reference.pt").exists()
    else selected
)
assert scoring_selection["train_step"] == scoring_selection["trainer_version"] == 600
assert {tag["weight_version"] for tag in scoring_selection["tags"]} == {600}
repeat = read("scheduling-repeats.pt")
isolation = json.loads((root / "scheduling-isolation.json").read_text())
assert isolation["max_foreign_requests_during_measurement"] == 0, isolation
assert set(repeat["scores"]) == {"serial1", "microbatch4", "concurrent32"}
assert all(len(runs) == 3 for runs in repeat["scores"].values())
data = scoring_selection["data"]
masks, policies = [], []
for index, tokens in enumerate(repeat["tokens"]):
    n = len(tokens)
    torch.testing.assert_close(
        torch.tensor(tokens), data["input_ids"].unbind()[index][:n], rtol=0, atol=0
    )
    masks.append(data["token_mask"].unbind()[index][:n].bool())
    policies.append(data["prev_logprobs"].unbind()[index][:n])


def compare(left, right):
    assert len(left) == len(right) == len(masks) == 32
    deltas = []
    for a, b, mask in zip(left, right, masks, strict=True):
        assert a.shape == b.shape == mask.shape
        deltas.append((a - b)[mask])
    return stats(deltas)


metrics = {}
for mode, runs in repeat["scores"].items():
    metrics[mode] = {
        "repeat_pairs": {
            f"{i}-{j}": compare(runs[j], runs[i])
            for i, j in itertools.combinations(range(3), 2)
        },
        "policy_comparisons": [compare(policies, run) for run in runs],
    }
cross = {
    f"{a}_vs_{b}": [
        compare(repeat["scores"][a][i], repeat["scores"][b][i]) for i in range(3)
    ]
    for a, b in itertools.combinations(repeat["scores"], 2)
}
residents = []
for rank in range(8):
    before = read(f"repeat-weights-before-{rank}.pt")
    after = read(f"repeat-weights-after-{rank}.pt")
    assert before.keys() == after.keys()
    for name, expected in before.items():
        actual = after[name]
        assert (
            expected["shape"] == actual["shape"]
            and expected["dtype"] == actual["dtype"]
        )
        assert torch.equal(
            expected["values"].view(torch.uint8), actual["values"].view(torch.uint8)
        ), (rank, name)
    residents.append(
        {"rank": rank, "tensors": len(before), "sampled_bits_unchanged": True}
    )
assert len(list((root / "train").glob("*.pt"))) == 64
assert len(list((root / "drift").glob("*.pt"))) == 32
assert Counter(path.stem.split("-")[0] for path in (root / "train").glob("*.pt")) == {
    "600": 32,
    "601": 32,
}
assert Counter(path.stem.split("-")[0] for path in (root / "drift").glob("*.pt")) == {
    "0": 16,
    "1": 16,
}
second = read("selected-601.pt")
assert second["train_step"] == second["trainer_version"] == 601
assert len(second["ids"]) == len(set(second["ids"])) == 32
assert not set(second["ids"]).intersection(selected["ids"])
assert {tag["weight_version"] for tag in second["tags"]} == {600}
per_step = {
    step: {name: [] for name in ("train_eval", "generation_policy", "drift")}
    for step in (600, 601)
}
for path in sorted((root / "train").glob("*.pt")):
    step = int(path.stem.split("-")[0])
    item = read(path.relative_to(root))
    for current, previous, generation, mask in zip(
        item["current"].unbind(),
        item["prev"].unbind(),
        item["generation"].unbind(),
        item["mask"].unbind(),
        strict=True,
    ):
        assert current.shape == previous.shape == generation.shape == mask.shape
        per_step[step]["train_eval"].append((current - previous)[mask.bool()])
        per_step[step]["generation_policy"].append((previous - generation)[mask.bool()])
for path in sorted((root / "drift").glob("*.pt")):
    step = 600 + int(path.stem.split("-")[0])
    item = read(path.relative_to(root))
    for before, after, token_mask in zip(
        item["before"].unbind(),
        item["after"].unbind(),
        item["data"]["token_mask"].unbind(),
        strict=True,
    ):
        length = token_mask.numel()
        before, after, mask = before[1:length], after[1:length], token_mask[1:].bool()
        assert before.shape == after.shape == mask.shape
        per_step[step]["drift"].append((after - before)[mask])
refits = []
for index in range(3):
    sender = read(f"sender-{index}.pt")
    for rank in range(8):
        receiver = read(f"receiver-{index}-{rank}.pt")
        assert sender.keys() == receiver.keys()
        for name, expected in sender.items():
            actual = receiver[name]
            assert (
                expected["shape"] == actual["shape"]
                and expected["dtype"] == actual["dtype"]
            )
            torch.testing.assert_close(
                expected["values"], actual["values"], rtol=0, atol=0
            )
    refits.append(
        {
            "refit": index,
            "tensors": len(sender),
            "receiver_ranks": 8,
            "payload_samples": "exact_match",
        }
    )
wire = [
    json.loads(line)
    for line in (root / "ssm-wire-dtype.jsonl").read_text().splitlines()
]
a_logs = [item for item in wire if item["name"].endswith(".A_log")]
assert len({item["name"] for item in a_logs}) == 30
assert all(
    item["wire"] == "torch.float32" and item["max_abs_error"] == 0 for item in a_logs
)
report = {
    "status": "completed",
    "scope": "Checkpoint600, unchanged FP32-A_log model, identical tokens; concurrency changes request scheduling, batch shapes and possible prefix-cache reuse together, not an isolated GEMM-kernel test.",
    "within_mode": metrics,
    "background_isolation": isolation,
    "cross_mode": cross,
    "resident_parameter_checks": residents,
    "refit_transport": refits,
    "fp32_a_log_wire_verified": True,
    "per_step": {
        str(step): {name: stats(values) for name, values in metrics.items()}
        for step, metrics in per_step.items()
    },
    "admission": [
        {
            "trainer_version": step,
            "rollout_versions": [600],
            "lag": step - 600,
            "unique_samples": 32,
        }
        for step in (600, 601)
    ],
    "pass_order": repeat["order"],
    "pass_seconds": repeat["seconds"],
    "actual_sampling_profile": json.loads((root / "sampling-profile.json").read_text()),
    "actual_execution_profile": [
        {
            "rank": item["rank"],
            "model_training": item["model_training"],
            "active_dropout_modules": item["active_dropout_modules"],
            "batch_invariance": item.get("batch_invariance", {"recorded": False}),
            "cache_scheduler": item.get("cache_scheduler", {"recorded": False}),
            "ssm_cache_dtypes": sorted(
                {layer["ssm_cache_dtype"] for layer in item["layers"]}
            ),
            "prefill_backends": sorted(
                {layer["prefill_backend"] for layer in item["layers"]}
            ),
        }
        for item in json.loads((root / "execution-profile.json").read_text())
    ],
}
reference_path = root / "scoring-reference.json"
if reference_path.exists():
    reference = json.loads(reference_path.read_text())
    source = Path(reference["source"])
    source_repeats = torch.load(
        source / "scheduling-repeats.pt", map_location="cpu", weights_only=False
    )
    assert repeat["tokens"] == source_repeats["tokens"]
    expected = torch.load(
        source / "sender-0.pt", map_location="cpu", weights_only=False
    )
    actual = read("sender-0.pt")
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert tensor["shape"] == actual[name]["shape"]
        assert tensor["dtype"] == actual[name]["dtype"]
        torch.testing.assert_close(
            tensor["values"], actual[name]["values"], rtol=0, atol=0
        )
    assert all(
        not item["batch_invariance"]["env_enabled"]
        and not item["cache_scheduler"]["enable_prefix_caching"]
        and item["cache_scheduler"]["enable_chunked_prefill"]
        and item["cache_scheduler"]["max_num_batched_tokens"] == 2048
        for item in report["actual_execution_profile"]
    )
    report["prefix_cache_disabled_vs_original_same_tokens"] = {
        "reference": reference,
        "initial_refit_hf_tensors_sampled_exact_match": len(expected),
        "scope": "Prefix cache disabled, unchanged2048 chunk budget; prompt-logprob requests already skip cache reads. This changes cache management and GDN alignment, not only prefix reuse.",
        "matched_concurrency_pairs": {
            mode: [
                compare(repeat["scores"][mode][i], source_repeats["scores"][mode][i])
                for i in range(3)
            ]
            for mode in repeat["scores"]
        },
    }
(root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
