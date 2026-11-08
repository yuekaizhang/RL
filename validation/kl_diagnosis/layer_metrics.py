"""Torch-only tensor comparison shared by isolated policy/generation environments."""

import torch


def compare(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """Compare equal-shaped finite tensors in FP64 without changing either."""
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    delta = actual.double() - reference.double()
    rms = delta.square().mean().sqrt().item()
    base = reference.double().square().mean().sqrt().item()
    return {
        "shape": list(actual.shape),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(reference.dtype),
        "relative_rms_delta": rms / base if base else None,
        "max_abs_delta": delta.abs().max().item(),
        "nonidentical_fraction": (delta != 0).double().mean().item(),
    }
