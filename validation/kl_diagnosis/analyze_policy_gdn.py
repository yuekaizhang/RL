"""Resolve AutoModel's actual GDN primitive and replay captured inputs in allocation."""

import inspect
import json
import math
import os
from pathlib import Path
import sys

import torch
from transformers import AutoConfig

from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
    CPAwareGatedDeltaNet,
)

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
root = Path(sys.argv[1])
capture = torch.load(root / "gdn-prefill-inputs.pt", weights_only=False)
inputs = capture["tensors"]
assert not capture["use_qk_l2norm_in_kernel"]
boundaries = inputs["cu_seqlens"].tolist()
lengths = [b - a for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)]
sequence = max(range(len(lengths)), key=lengths.__getitem__)
start, length = boundaries[sequence], min(lengths[sequence], 1024)
values = {
    key: inputs[key][:, start : start + length].contiguous()
    for key in ("q", "k", "v", "g", "beta")
}
initial = inputs["initial_state"][sequence : sequence + 1].contiguous()
assert initial.abs().max().item() == 0


@torch.no_grad()
def cpu_reference(dtype):
    q, k, v, g, beta = (
        values[key][0].to(dtype) for key in ("q", "k", "v", "g", "beta")
    )
    value_heads = v.shape[1]
    q = q.repeat_interleave(value_heads // q.shape[1], dim=1) / math.sqrt(q.shape[-1])
    k = k.repeat_interleave(value_heads // k.shape[1], dim=1)
    state = initial[0].to(dtype).clone()
    outputs = []
    for position in range(length):
        state *= g[position].exp()[:, None, None]
        delta = v[position] - (state * k[position, :, None, :]).sum(-1)
        delta *= beta[position, :, None]
        state += delta[:, :, None] * k[position, :, None, :]
        outputs.append((state * q[position, :, None, :]).sum(-1))
    return torch.stack(outputs).unsqueeze(0), state.unsqueeze(0)


def compare(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    delta = actual.double() - expected.double()
    rms = delta.square().mean().sqrt().item()
    base_rms = expected.double().square().mean().sqrt().item()
    return {
        "rms_delta": rms,
        "relative_rms_delta": rms / base_rms if base_rms else None,
        "max_abs_delta": delta.abs().max().item(),
        "nonidentical_fraction": (delta != 0).double().mean().item(),
    }


# Establish the higher precision CPU reference before calling any GPU kernel.
torch.set_num_threads(8)
reference32 = cpu_reference(torch.float32)
reference64 = cpu_reference(torch.float64)
torch.cuda.set_device(0)
config = AutoConfig.from_pretrained(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy/models/Qwen/Qwen3.5-35B-A3B-Base",
    trust_remote_code=True,
).text_config
torch.manual_seed(42)
# Only one untrained component is constructed to inspect its actual binding;
# its weights are never used. No checkpoint or full model is loaded.
layer = CPAwareGatedDeltaNet(config, layer_idx=0)
primitive = layer.chunk_gated_delta_rule
report = {
    "status": "in_progress",
    "scope": "AutoModel's resolved GDN primitive on one captured pre-normalized segment; not a full-layer/model parity test. BF16 beta is a source-motivated counterfactual, not a capture of the policy's projected gates.",
    "source": str(root / "gdn-prefill-inputs.pt"),
    "tokens": length,
    "primitive_module": primitive.__module__,
    "primitive_name": primitive.__name__,
    "primitive_source": inspect.getsourcefile(inspect.unwrap(primitive)),
    "primitive_module_file": inspect.getmodule(primitive).__file__,
    "primitive_doc": inspect.getdoc(primitive),
    "layer_class_source": inspect.getsourcefile(type(layer)),
    "cpu_fp32_vs_fp64_output": compare(reference32[0], reference64[0]),
    "cpu_fp32_vs_fp64_state": compare(reference32[1], reference64[1]),
    "cases": {},
}
expanded = {key: value.cuda() for key, value in values.items()}
heads = expanded["v"].shape[2] // expanded["q"].shape[2]
expanded["q"] = expanded["q"].repeat_interleave(heads, dim=2)
expanded["k"] = expanded["k"].repeat_interleave(heads, dim=2)
payload = {"inputs": values, "reference64": reference64, "cases": {}}


@torch.no_grad()
def run(beta_dtype):
    args = {key: value.clone() for key, value in expanded.items()}
    args["beta"] = args["beta"].to(beta_dtype)
    output, state = primitive(
        **args,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, length], device="cuda", dtype=torch.int32),
        use_qk_l2norm_in_kernel=False,
    )
    return output.detach().cpu(), state.detach().cpu()


for name, dtype in (("shared_beta_fp32", torch.float32), ("beta_bf16", torch.bfloat16)):
    run(dtype)
    runs = [run(dtype) for _ in range(3)]
    output, state = runs[0]
    payload["cases"][name] = (output, state)
    report["cases"][name] = {
        "repeat_output": [compare(x[0], output) for x in runs[1:]],
        "repeat_state": [compare(x[1], state) for x in runs[1:]],
        "output_vs_cpu_fp64": compare(output, reference64[0]),
        "state_shape": list(state.shape),
        "state_dtypes": str(state.dtype),
        # Record both orientations explicitly; use the resolved API/source to
        # interpret the state contract rather than silently choosing the closer.
        "state_as_vk_vs_cpu_fp64": compare(state, reference64[1]),
        "state_as_kv_vs_cpu_fp64": compare(state.transpose(-1, -2), reference64[1]),
    }
    torch.save(payload, root / "policy-gdn-replay.pt")
    (root / "policy-gdn-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
report["status"] = "completed"
(root / "policy-gdn-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
