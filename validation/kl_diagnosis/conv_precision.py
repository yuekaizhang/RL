"""Isolate causal-convolution product precision using saved first-layer inputs."""

import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from validation.kl_diagnosis.layer_metrics import compare


@torch.no_grad()
def reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    dtype: torch.dtype,
    *,
    round_products: bool = False,
) -> torch.Tensor:
    """Causal depthwise cross-correlation, zero left state, width4, no bias."""
    length, channels = x.shape
    assert weight.shape == (channels, 4)
    padded = F.pad(x.to(dtype), (0, 0, 3, 0))
    result = torch.zeros(length, channels, dtype=dtype)
    for tap in range(4):
        product = padded[tap : tap + length] * weight[:, tap].to(dtype)
        if round_products:
            product = product.bfloat16().to(dtype)
        result += product
    return result


@torch.no_grad()
def policy(root: Path) -> None:
    # Heavy policy-only imports stay in its isolated interpreter.
    from causal_conv1d import causal_conv1d_fn

    payload = torch.load(
        root / "layer-policy.pt", map_location="cpu", weights_only=False
    )
    x = payload["stages"]["qkv"].squeeze(0)
    weight = payload["weights"]["linear_attn.conv1d.weight"].squeeze(1)
    torch.set_num_threads(8)
    # Establish references before invoking GPU convolution kernels.
    pre64 = reference(x, weight, torch.float64)
    pre32 = reference(x, weight, torch.float32)
    pre_product_bf16 = reference(x, weight, torch.float32, round_products=True)
    references = {
        "pre64": pre64,
        "pre32": pre32,
        "pre_product_bf16": pre_product_bf16,
        "out64": F.silu(pre64),
        "out32": F.silu(pre32),
        "out_product_bf16": F.silu(pre_product_bf16),
        "out_pre_bf16": F.silu(pre64.bfloat16().double()),
    }
    x_gpu, weight_gpu = x.cuda().transpose(0, 1).unsqueeze(0), weight.cuda()
    outputs = {}
    for label, activation in (("pre", None), ("silu", "silu")):
        values = [
            causal_conv1d_fn(x_gpu, weight_gpu, activation=activation)
            .squeeze(0)
            .transpose(0, 1)
            .cpu()
            for _ in range(3)
        ]
        assert all(torch.equal(values[0], value) for value in values[1:])
        outputs[label] = values[0]
    assert torch.equal(outputs["silu"], payload["stages"]["conv"].squeeze(0))
    torch.save(
        {"references": references, "policy": outputs}, root / "conv-reference.pt"
    )
    print("CONV_POLICY_REFERENCE_COMPLETED", flush=True)


@torch.no_grad()
def generation(root: Path, source: Path) -> None:
    # Heavy generation-only imports stay in its isolated interpreter.
    from triton.runtime.jit import JITFunction
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as implementation
    from validation.kl_diagnosis.layer_generation import main as compare_layer

    payload = torch.load(
        root / "layer-policy.pt", map_location="cpu", weights_only=False
    )
    references = torch.load(
        root / "conv-reference.pt", map_location="cpu", weights_only=False
    )
    x = payload["stages"]["qkv"].squeeze(0).cuda()
    weight = payload["weights"]["linear_attn.conv1d.weight"].squeeze(1).cuda()
    cu = torch.tensor([0, x.shape[0]], dtype=torch.int32, device="cuda")

    def run(activation: str | None) -> torch.Tensor:
        return (
            implementation.causal_conv1d_fn(
                x.transpose(0, 1),
                weight,
                None,
                torch.zeros(2, x.shape[1], 3, device="cuda", dtype=torch.bfloat16),
                cu,
                cache_indices=torch.ones(1, device="cuda", dtype=torch.int32),
                has_initial_state=torch.zeros(1, device="cuda", dtype=torch.bool),
                activation=activation,
            )
            .transpose(0, 1)
            .cpu()
        )

    original = implementation._causal_conv1d_fwd_kernel
    before = {
        name: run(activation) for name, activation in (("pre", None), ("silu", "silu"))
    }
    target = "acc += matrix_x * matrix_w"
    replacement = "acc += matrix_x.to(tl.float32) * matrix_w.to(tl.float32)"
    assert original.src.count(target) == 1
    # New JITFunction => fresh specialization cache. Never mutate the existing
    # cached kernel or the image's files. The Python caller looks up this global.
    modified = JITFunction(
        original.fn, do_not_specialize_on_alignment=["num_cache_lines"]
    )
    modified._unsafe_update_src(original.src.replace(target, replacement))
    report = {
        "status": "in_progress",
        "scope": "Same saved first-layer projected BF16 inputs/weights; only explicit FP32 product casts in the pinned vLLM prefill-conv JIT. Local diagnostic replacement restored afterwards; no production/image/checkpoint modification or full-model KL.",
        "source_component": str(source),
        "patch": {"before": target, "after": replacement},
        "cpu_fp32_vs_fp64_pre": compare(
            references["references"]["pre32"], references["references"]["pre64"]
        ),
        "cases": {},
    }
    try:
        implementation._causal_conv1d_fwd_kernel = modified
        after = {
            name: run(activation)
            for name, activation in (("pre", None), ("silu", "silu"))
        }
        for activation in (None, "silu"):
            expected = after["pre" if activation is None else "silu"]
            assert all(torch.equal(run(activation), expected) for _ in range(2))
        for name in ("pre", "silu"):
            ref = references["references"]["pre64" if name == "pre" else "out64"]
            rounded_product = references["references"][
                "pre_product_bf16" if name == "pre" else "out_product_bf16"
            ]
            for label, value in (
                ("policy", references["policy"][name]),
                ("vllm_original", before[name]),
                ("vllm_fp32_product", after[name]),
            ):
                report["cases"][label + "_" + name] = {
                    "vs_fp64": compare(value, ref),
                    "vs_fp64_rounded_bf16": compare(value, ref.bfloat16()),
                    "vs_bf16_product_reference_rounded_bf16": compare(
                        value, rounded_product.bfloat16()
                    ),
                    "vs_policy": compare(value, references["policy"][name]),
                }
        (root / "conv-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
        # Reuse the complete component comparison with this one changed stage.
        compare_layer(root)
        baseline = json.loads((source / "layer-analysis.json").read_text())
        changed = json.loads((root / "layer-analysis.json").read_text())
        report["component_output_vs_policy_before_after"] = {
            label: {
                "before": baseline["pipelines"][label]["versus_policy"]["output"],
                "after": result["versus_policy"]["output"],
            }
            for label, result in changed["pipelines"].items()
        }
    finally:
        implementation._causal_conv1d_fwd_kernel = original
    restored = run("silu")
    assert torch.equal(restored, before["silu"])
    report["restored_original_output"] = compare(restored, before["silu"])
    report["status"] = "completed"
    torch.save({"before": before, "after": after}, root / "conv-generation.pt")
    (root / "conv-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CONV_PRECISION_COMPLETED", json.dumps(report), flush=True)


if __name__ == "__main__":
    assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
    torch.cuda.set_device(0)
    task_root = Path(sys.argv[2])
    if sys.argv[1] == "policy":
        policy(task_root)
    else:
        assert sys.argv[1] == "generation"
        generation(task_root, Path(sys.argv[3]))
