"""GPU regression for probing tensors larger than float32's exact integer range."""

import json
import os
from pathlib import Path

import torch

from validation.kl_diagnosis.sampling import sample_indices, sample_tensor

assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
width = 248320 * 2048
old = torch.linspace(0, width - 1, 64, device="cuda", dtype=torch.float32).long()
old_last = old[-1].item()
assert old_last >= width, (old_last, width)
for size in (0, 1, 17, 64, 65, 2**24, 2**24 + 1, width, 256 * 2048 * 2048):
    indices = sample_indices(size, "cuda").cpu().tolist()
    expected = [
        i * (size - 1) // max(min(64, size) - 1, 1) for i in range(min(64, size))
    ]
    assert indices == expected, (size, indices, expected)
    assert all(0 <= value < size for value in indices)
    if indices:
        assert indices[0] == 0 and indices[-1] == size - 1
experts = torch.arange(2 * 3 * 5, device="cuda", dtype=torch.bfloat16).reshape(2, 3, 5)
sampled = sample_tensor(experts)
torch.testing.assert_close(
    sampled["values"], experts.reshape(2, 15).cpu(), rtol=0, atol=0
)
assert sample_tensor(torch.empty((0, 2, 3), device="cuda"))["values"].numel() == 0
report = {
    "status": "passed",
    "embedding_elements": width,
    "old_last_index": old_last,
    "valid_last_index": width - 1,
    "new_indices_in_bounds": True,
    "all_experts_checked": True,
}
root = Path("validation/kl_diagnosis/results")
(root / f"sampling-check-{os.environ['SLURM_JOB_ID']}.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2), flush=True)
