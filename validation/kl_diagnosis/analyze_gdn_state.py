"""Replay actual GDN inputs to isolate recurrent-state storage rounding."""

import json
import os
import sys
from pathlib import Path

import torch

from vllm.third_party.flash_linear_attention.ops import fused_recurrent_gated_delta_rule

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
root = Path(sys.argv[1])
capture = torch.load(
    root / "gdn-prefill-inputs.pt", map_location="cpu", weights_only=False
)
inputs = capture["tensors"]
boundaries = inputs["cu_seqlens"].tolist()
lengths = [
    end - start for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
]
sequence = max(range(len(lengths)), key=lengths.__getitem__)
start = boundaries[sequence]
length = min(lengths[sequence], 1024)
assert length > 0 and inputs["initial_state"].shape[0] == len(lengths)
torch.cuda.set_device(0)
tensors = {
    key: inputs[key][:, start : start + length].contiguous().cuda()
    for key in ("q", "k", "v", "g", "beta")
}
initial = inputs["initial_state"][sequence : sequence + 1].contiguous().cuda().float()


@torch.no_grad()
def replay(storage_dtype, one_token_at_a_time):
    # This pinned vLLM kernel reserves state slot0 and requires explicit valid
    # indices for in-place writes, even though its public signature permits None.
    state = torch.cat((torch.zeros_like(initial), initial)).to(storage_dtype).clone()
    indices = torch.ones((1, length), device="cuda", dtype=torch.int32)
    outputs = []
    for position in range(length) if one_token_at_a_time else (None,):
        args = {
            key: value[:, position : position + 1].contiguous()
            if position is not None
            else value
            for key, value in tensors.items()
        }
        output, final = fused_recurrent_gated_delta_rule(
            **args,
            initial_state=state,
            inplace_final_state=True,
            ssm_state_indices=indices[:, :1] if position is not None else indices,
            use_qk_l2norm_in_kernel=capture["use_qk_l2norm_in_kernel"],
        )
        assert final.data_ptr() == state.data_ptr()
        outputs.append(output)
    result = torch.cat(outputs, dim=1)
    assert result.shape == (1, length, tensors["v"].shape[2], tensors["v"].shape[3])
    assert torch.isfinite(result).all() and torch.isfinite(state).all()
    return result.float(), state[1:].float()


def compare(actual, reference):
    delta = (actual - reference).double()
    rms = delta.square().mean().sqrt().item()
    reference_rms = reference.double().square().mean().sqrt().item()
    return {
        "elements": delta.numel(),
        "rms_delta": rms,
        "max_abs_delta": delta.abs().max().item(),
        "relative_rms_delta": rms / reference_rms if reference_rms else None,
        "nonidentical_fraction": (delta != 0).double().mean().item(),
    }


full_output, full_state = replay(torch.float32, False)
step_fp32_output, step_fp32_state = replay(torch.float32, True)
step_bf16_output, step_bf16_state = replay(torch.bfloat16, True)
report = {
    "status": "completed",
    "scope": "One actual GDN layer, identical captured q/k/v/g/beta/initial state, at most 1024 tokens of one prefill segment. Uses the pinned Triton recurrent kernel; not full-model KL or a CUDA decode-kernel verdict.",
    "layer": capture["layer"],
    "sequence_index": sequence,
    "segment_tokens": lengths[sequence],
    "replayed_tokens": length,
    "captured_state_dtype": str(inputs["initial_state"].dtype),
    "initial_state_bf16_rounding_max_abs": (initial - initial.bfloat16().float())
    .abs()
    .max()
    .item(),
    "fp32_per_token_vs_fp32_continuous_output": compare(step_fp32_output, full_output),
    "bf16_per_token_vs_fp32_per_token_output": compare(
        step_bf16_output, step_fp32_output
    ),
    "bf16_per_token_vs_fp32_continuous_output": compare(step_bf16_output, full_output),
    "fp32_per_token_vs_fp32_continuous_state": compare(step_fp32_state, full_state),
    "bf16_per_token_vs_fp32_per_token_state": compare(step_bf16_state, step_fp32_state),
}
(root / "gdn-state-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
