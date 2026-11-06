"""Analyze fixed-token logprob and refit probes inside an allocation."""

import json
import os
import sys
from pathlib import Path

import torch

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
root = Path(sys.argv[1])

admission = []
all_ids = set()
for step in (600, 601):
    selection = torch.load(
        root / f"selected-{step}.pt", map_location="cpu", weights_only=False
    )
    assert selection["train_step"] == selection["trainer_version"] == step
    assert len(selection["ids"]) == len(set(selection["ids"])) == 32
    assert not all_ids.intersection(selection["ids"])
    all_ids.update(selection["ids"])
    versions = [tag["weight_version"] for tag in selection["tags"]]
    assert len(versions) == 32 and set(versions) == {600}, versions
    # Recovery redispatches unfinished groups from checkpoint600. The first
    # batch is same-version; the second is lag1, even with diagnostic LA0.
    lags = sorted({step - version for version in versions})
    admission.append(
        {
            "step": step,
            "trainer_version": selection["trainer_version"],
            "rollout_versions": sorted(set(versions)),
            "lag": lags,
            "unique_samples": 32,
        }
    )


def stats(values):
    x = torch.cat(values).double()
    assert torch.isfinite(x).all()
    return {
        "tokens": x.numel(),
        "mean_logprob_delta": x.mean().item(),
        "rms_logprob_delta": x.square().mean().sqrt().item(),
        "max_abs_delta": x.abs().max().item(),
        "positive_fraction": (x > 0).double().mean().item(),
        "negative_fraction": (x < 0).double().mean().item(),
        "mean_k3": (x.exp() - 1 - x).mean().item(),
        "reverse_mean_k3": ((-x).exp() - 1 + x).mean().item(),
    }


train_eval, gen_policy, drift = [], [], []
per_step = {
    step: {"train_eval": [], "generation_policy": [], "drift": []}
    for step in (600, 601)
}
train_files = sorted((root / "train").glob("*.pt"))
drift_files = sorted((root / "drift").glob("*.pt"))
assert len(train_files) == 64, len(train_files)
assert len(drift_files) == 32, len(drift_files)
for path in train_files:
    step = int(path.stem.split("-")[0])
    d = torch.load(path, map_location="cpu", weights_only=False)
    for current, prev, generation, mask in zip(
        d["current"].unbind(),
        d["prev"].unbind(),
        d["generation"].unbind(),
        d["mask"].unbind(),
        strict=True,
    ):
        assert current.shape == prev.shape == generation.shape == mask.shape
        train_eval.append((current - prev)[mask.bool()])
        gen_policy.append((prev - generation)[mask.bool()])
        per_step[step]["train_eval"].append((current - prev)[mask.bool()])
        per_step[step]["generation_policy"].append((prev - generation)[mask.bool()])
for path in drift_files:
    step = 600 + int(path.stem.split("-")[0])
    d = torch.load(path, map_location="cpu", weights_only=False)
    for before, after, token_mask in zip(
        d["before"].unbind(),
        d["after"].unbind(),
        d["data"]["token_mask"].unbind(),
        strict=True,
    ):
        n = token_mask.numel()
        mask = token_mask[1:].bool()
        before, after = before[1:n], after[1:n]
        assert before.shape == after.shape == mask.shape
        drift.append((after - before)[mask])
        per_step[step]["drift"].append((after - before)[mask])

weights = []
for sender_path in sorted(root.glob("sender-*.pt")):
    version = sender_path.stem.split("-")[1]
    sender = torch.load(sender_path, map_location="cpu", weights_only=False)
    receivers = sorted(root.glob(f"receiver-{version}-*.pt"))
    assert len(receivers) == 8, (version, len(receivers))
    for path in receivers:
        receiver = torch.load(path, map_location="cpu", weights_only=False)
        assert sender.keys() == receiver.keys(), (
            version,
            sender.keys() - receiver.keys(),
        )
        for name, expected in sender.items():
            actual = receiver[name]
            assert (
                expected["shape"] == actual["shape"]
                and expected["dtype"] == actual["dtype"]
            )
            torch.testing.assert_close(
                expected["values"], actual["values"], rtol=0, atol=0
            )
    weights.append(
        {
            "refit": int(version),
            "tensors": len(sender),
            "receiver_ranks": 8,
            "payload_samples": "exact_match",
        }
    )
assert len(weights) >= 2, weights
report = {
    "status": "completed",
    "scope": "AutoModel checkpoint600, EP16, configured LA0, two real updates; recovered batches have lag0 then lag1. Sampled refit payloads, not exhaustive resident mapping proof.",
    "train_eval_same_version": stats(train_eval),
    "generation_policy_mixed_lag0_lag1": stats(gen_policy),
    "policy_single_update_same_tokens": stats(drift),
    "refit_transport": weights,
    "admission": admission,
    "per_step": {
        str(step): {name: stats(values) for name, values in metrics.items()}
        for step, metrics in per_step.items()
    },
}
ab = torch.load(root / "fixed-token-ssm-ab.pt", map_location="cpu", weights_only=False)
selected = torch.load(root / "selected-600.pt", map_location="cpu", weights_only=False)[
    "data"
]
prefill_old, prefill_fp32, prefill_decode = [], [], []
controls = {
    key: [] for key in ("before_repeat", "after_repeat", "restored_before") if key in ab
}
for i, (before, after) in enumerate(zip(ab["before"], ab["after"], strict=True)):
    n = len(before)
    assert len(after) == len(ab["tokens"][i]) == n
    torch.testing.assert_close(
        torch.tensor(ab["tokens"][i]),
        selected["input_ids"][i][:n],
        rtol=0,
        atol=0,
    )
    mask = selected["token_mask"].unbind()[i][:n].bool()
    policy = selected["prev_logprobs"].unbind()[i][:n]
    generation = selected["generation_logprobs"].unbind()[i][:n]
    prefill_old.append((policy - before)[mask])
    prefill_fp32.append((policy - after)[mask])
    prefill_decode.append((before - generation)[mask])
    for key in controls:
        reference = after if key == "after_repeat" else before
        actual = ab[key][i]
        assert actual.shape == reference.shape
        controls[key].append((actual - reference)[mask])
report["a_log_fp32_same_tokens_same_policy"] = {
    "original_refit_prefill_vs_policy": stats(prefill_old),
    "fp32_a_log_prefill_vs_policy": stats(prefill_fp32),
    "original_prefill_vs_original_decode": stats(prefill_decode),
    "changes": ab["changes"],
    "restored": ab["restored"],
}
if controls:
    assert len(controls) == 3
    report["a_log_repeat_controls"] = {
        "legacy_a_log_override": ab["legacy_a_log_override"],
        "same_weights_same_tokens": {
            key: stats(values) for key, values in controls.items()
        },
    }
kernel_path = root / "prefill-kernel-ab.pt"
if kernel_path.exists():
    kernel = torch.load(kernel_path, map_location="cpu", weights_only=False)
    assert kernel["tokens"] == ab["tokens"]
    assert len(kernel["switched"]) == len(kernel["restored"]) == 8
    assert all(item["triton_layers"] == 30 for item in kernel["switched"])
    assert all(item["restored_layers"] == 30 for item in kernel["restored"])
    assert all(
        len(item["triton_calls"]) == 30 and min(item["triton_calls"].values()) > 0
        for item in kernel["restored"]
    ), "The alternate method must actually run on every layer/rank"
    comparisons = {
        name: []
        for name in (
            "flashinfer_vs_policy",
            "triton_vs_policy",
            "flashinfer_vs_decode",
            "triton_vs_decode",
            "triton_vs_flashinfer",
            "triton_repeat",
            "flashinfer_restored",
        )
    }
    for i, tokens in enumerate(kernel["tokens"]):
        n = len(tokens)
        mask = selected["token_mask"].unbind()[i][:n].bool()
        policy = selected["prev_logprobs"].unbind()[i][:n]
        generation = selected["generation_logprobs"].unbind()[i][:n]
        fi, triton = kernel["flashinfer"][i], kernel["triton"][i]
        assert fi.shape == triton.shape == policy.shape
        deltas = {
            "flashinfer_vs_policy": policy - fi,
            "triton_vs_policy": policy - triton,
            "flashinfer_vs_decode": fi - generation,
            "triton_vs_decode": triton - generation,
            "triton_vs_flashinfer": triton - fi,
            "triton_repeat": kernel["triton_repeat"][i] - triton,
            "flashinfer_restored": kernel["flashinfer_restored"][i] - fi,
        }
        for name, delta in deltas.items():
            comparisons[name].append(delta[mask])
    report["prefill_kernel_ab"] = {
        "scope": "Identical checkpoint600 policy/tokens/weights, restored legacy BF16-rounded A_log; only prefill method changes. Original decode is CUDA.",
        "metrics": {name: stats(values) for name, values in comparisons.items()},
        "switched": kernel["switched"],
        "restored": [
            {
                "rank": item["rank"],
                "restored_layers": item["restored_layers"],
                "min_triton_calls_per_layer": min(item["triton_calls"].values()),
                "total_triton_calls": sum(item["triton_calls"].values()),
            }
            for item in kernel["restored"]
        ],
    }
for filename, key in (
    ("sampling-profile.json", "actual_sampling_profile"),
    ("execution-profile.json", "actual_execution_profile"),
):
    path = root / filename
    if path.exists():
        profile = json.loads(path.read_text())
        if key == "actual_execution_profile":
            profile = [
                {
                    "rank": item["rank"],
                    "layers": len(item["layers"]),
                    **{
                        name: sorted({layer[name] for layer in item["layers"]})
                        for name in (
                            "prefill_backend",
                            "prefill_method",
                            "decode_kernel",
                            "conv_cache_dtype",
                            "ssm_cache_dtype",
                            "a_log_dtype",
                            "dt_bias_dtype",
                        )
                    },
                }
                for item in profile
            ]
        report[key] = profile
(root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
