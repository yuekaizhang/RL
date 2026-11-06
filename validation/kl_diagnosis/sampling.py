"""Tensor-only refit probe shared by isolated policy and generation environments."""

import torch


def sample_indices(width, device):
    count = min(64, width)
    # Float linspace rounds width-1 up to width for large model tensors.
    # Keep all arithmetic integer so the last sample is always in bounds.
    return (
        torch.arange(count, device=device, dtype=torch.int64)
        * (width - 1)
        // max(count - 1, 1)
    )


def sample_tensor(tensor):
    tensor = tensor.detach()
    if tensor.numel() == 0:
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "values": torch.empty(0, dtype=tensor.dtype),
        }
    # For expert stacks probe every expert, rather than only the first expert.
    matrix = (
        tensor.reshape(tensor.shape[0], -1)
        if tensor.ndim == 3
        else tensor.reshape(1, -1)
    )
    indices = sample_indices(matrix.shape[-1], tensor.device)
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "values": matrix[:, indices].cpu().clone(),
    }
