"""Numerical helpers; these do not add learned fusion mechanisms."""

import torch


def autocast_enabled(device):
    """Query the active autocast context for the input's actual device."""
    return torch.is_autocast_enabled(device.type)


def validate_tensor(name, tensor, *, probability=False):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != 4 or any(size < 1 for size in tensor.shape):
        raise ValueError(f"{name} must be a non-empty [B,C,H,W] tensor.")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating-point.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    if probability and ((tensor < 0).any() or (tensor > 1).any()):
        raise ValueError(f"{name} must lie in [0,1].")


def align_autocast_inputs(module, *tensors):
    """Allow AMP outputs from preceding modules to meet fp32 parameters.

    Outside autocast, callers still enforce their strict dtype contracts.
    Casts remain differentiable and never mutate input tensors or parameters.
    """
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return tensors  # Let the public validator give its normal TypeError.
    if autocast_enabled(tensors[0].device):
        dtype = next(module.parameters()).dtype
        return tuple(t.to(dtype=dtype) for t in tensors)
    return tensors
