"""Replay one captured GDN segment through the pinned kernels and FP64 math."""

import json
import math
import os
from pathlib import Path
import sys

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fi_chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
root = Path(sys.argv[1])
capture = torch.load(
    root / "gdn-prefill-inputs.pt", map_location="cpu", weights_only=False
)
inputs = capture["tensors"]
boundaries = inputs["cu_seqlens"].tolist()
lengths = [b - a for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)]
sequence = max(range(len(lengths)), key=lengths.__getitem__)
start, length = boundaries[sequence], min(lengths[sequence], 1024)
assert length > 0 and inputs["initial_state"].shape[0] == len(lengths)
torch.cuda.set_device(0)
tensors = {
    key: inputs[key][:, start : start + length].contiguous().cuda()
    for key in ("q", "k", "v", "g", "beta")
}
initial = inputs["initial_state"][sequence : sequence + 1].contiguous().cuda()
assert initial.dtype == torch.float32
normalize = capture["use_qk_l2norm_in_kernel"]
cu_seqlens = torch.tensor([0, length], device="cuda", dtype=torch.int32)


def compare(actual, reference):
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    delta = actual.double() - reference.double()
    rms = delta.square().mean().sqrt().item()
    reference_rms = reference.double().square().mean().sqrt().item()
    return {
        "elements": delta.numel(),
        "rms_delta": rms,
        "relative_rms_delta": rms / reference_rms if reference_rms else None,
        "max_abs_delta": delta.abs().max().item(),
        "nonidentical_fraction": (delta != 0).double().mean().item(),
    }


@torch.no_grad()
def run_chunk(backend, values, qk_normalize):
    fn = (
        fi_chunk_gated_delta_rule if backend == "flashinfer" else chunk_gated_delta_rule
    )
    output, state = fn(
        **{key: value.clone() for key, value in values.items()},
        initial_state=initial.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=qk_normalize,
    )
    return output.clone(), state.clone()


@torch.no_grad()
def run_recurrent(values):
    state = torch.cat((torch.zeros_like(initial), initial)).clone()
    output, final = fused_recurrent_gated_delta_rule(
        **{key: value.clone() for key, value in values.items()},
        initial_state=state,
        inplace_final_state=True,
        ssm_state_indices=torch.ones((1, length), device="cuda", dtype=torch.int32),
        use_qk_l2norm_in_kernel=False,
    )
    assert final.data_ptr() == state.data_ptr()
    return output.clone(), state[1:].clone()


@torch.no_grad()
def reference_fp64(values):
    """Sequential delta rule on shared, already normalized q/k; no BF16 state writes."""
    q, k, v, g, beta = (values[key][0].double() for key in ("q", "k", "v", "g", "beta"))
    query_heads, value_heads = q.shape[1], v.shape[1]
    assert value_heads % query_heads == 0
    q = q.repeat_interleave(value_heads // query_heads, dim=1) / math.sqrt(q.shape[-1])
    k = k.repeat_interleave(value_heads // query_heads, dim=1)
    state = initial[0].double().clone()
    assert state.shape == (value_heads, v.shape[-1], k.shape[-1])
    outputs = []
    for position in range(length):
        state *= g[position].exp()[:, None, None]
        delta = v[position] - (state * k[position, :, None, :]).sum(-1)
        weight = beta[position]
        delta *= weight[:, None] if weight.ndim == 1 else weight
        state += delta[:, :, None] * k[position, :, None, :]
        outputs.append((state * q[position, :, None, :]).sum(-1))
    return torch.stack(outputs).unsqueeze(0), state.unsqueeze(0)


report = {
    "status": "in_progress",
    "scope": "One actual captured GDN layer/segment, at most1024 tokens; compares kernel output and state, not full-model KL or CUDA decode correctness. FP64 reference uses identical captured inputs and shared BF16-normalized q/k with the sequential delta rule.",
    "layer": capture["layer"],
    "source_capture": str(root / "gdn-prefill-inputs.pt"),
    "sequence_index": sequence,
    "segment_tokens": lengths[sequence],
    "replayed_tokens": length,
    "input_shapes": {key: list(value.shape) for key, value in tensors.items()},
    "input_dtypes": {key: str(value.dtype) for key, value in tensors.items()},
    "initial_state_dtype": str(initial.dtype),
    "initial_state_max_abs": initial.abs().max().item(),
    "qk_normalize_in_captured_call": normalize,
    "gpu": torch.cuda.get_device_name(),
    "torch_version": torch.__version__,
    "original_paths": {},
}
path = root / "gdn-kernel-analysis.json"


def save():
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


original = {}
for backend in ("flashinfer", "triton"):
    run_chunk(backend, tensors, normalize)
    runs = [run_chunk(backend, tensors, normalize) for _ in range(3)]
    original[backend] = runs[0]
    report["original_paths"][backend] = {
        "repeat_output": [compare(item[0], runs[0][0]) for item in runs[1:]],
        "repeat_state": [compare(item[1], runs[0][1]) for item in runs[1:]],
    }
    save()
report["flashinfer_vs_triton_original"] = {
    "output": compare(original["flashinfer"][0], original["triton"][0]),
    "state": compare(original["flashinfer"][1], original["triton"][1]),
}
shared = dict(tensors)
if normalize:
    shared["q"], shared["k"] = l2norm_fwd(tensors["q"]), l2norm_fwd(tensors["k"])
reference_output, reference_state = reference_fp64(shared)
report["shared_normalization"] = {}
for backend in ("flashinfer", "triton", "recurrent_fp32"):
    output, state = (
        run_recurrent(shared)
        if backend == "recurrent_fp32"
        else run_chunk(backend, shared, False)
    )
    report["shared_normalization"][backend] = {
        "output_vs_fp64": compare(output, reference_output),
        "state_vs_fp64": compare(state, reference_state),
    }
    if backend in original:
        report["shared_normalization"][backend]["output_vs_original"] = compare(
            output, original[backend][0]
        )
    save()
report["reference_output_rounding_floor"] = compare(
    reference_output.to(tensors["v"].dtype), reference_output
)
policy_path = root / "policy-gdn-replay.pt"
if policy_path.exists():
    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    for key, actual in tensors.items():
        torch.testing.assert_close(actual.cpu(), policy["inputs"][key], rtol=0, atol=0)
    expanded = dict(shared)
    heads = shared["v"].shape[2] // shared["q"].shape[2]
    expanded["q"] = shared["q"].repeat_interleave(heads, dim=2)
    expanded["k"] = shared["k"].repeat_interleave(heads, dim=2)
    report["policy_primitive_parity"] = {
        "scope": "Exact shared q/k/v/g/beta input values across policy and generation environments; q/k heads explicitly expanded as in AutoModel. BF16 beta is a separate counterfactual.",
        "gpu_fp64_vs_cpu_fp64_output": compare(
            reference_output.cpu(), policy["reference64"][0]
        ),
        "gpu_fp64_vs_cpu_fp64_state": compare(
            reference_state.cpu(), policy["reference64"][1]
        ),
        "kernels": {},
    }
    for backend in ("flashinfer", "triton"):
        output, _ = run_chunk(backend, expanded, False)
        report["policy_primitive_parity"]["kernels"][backend] = {
            "expanded_heads_vs_original_output": compare(output, original[backend][0]),
            "vs_policy_shared_beta_fp32_output": compare(
                output.cpu(), policy["cases"]["shared_beta_fp32"][0]
            ),
            "vs_policy_beta_bf16_output": compare(
                output.cpu(), policy["cases"]["beta_bf16"][0]
            ),
        }
report["status"] = "completed"
save()
