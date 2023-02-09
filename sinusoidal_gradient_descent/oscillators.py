"""Decay oscillator variants"""

import torch


def complex_oscillator_direct(z: torch.complex, N: int):
    """Implements the complex surrogate by direct exponentiation."""
    n = torch.arange(N)
    return (z ** n).real


def complex_oscillator_cumprod(z: torch.complex, N: int):
    """Implements the complex surrogate by cumulative product of z."""
    z = z.repeat(N)
    initial = torch.ones(*z.shape[:-1], 1, dtype=z.dtype, device=z.device)
    z_cat = torch.cat([initial, z], dim=-1)[:-1]
    return torch.cumprod(z_cat, dim=-1).real


def complex_oscillator_damped(z: torch.complex, N: int):
    """Implements the complex surrogate by explicit damped sinusoid parameters."""
    n = torch.arange(N)
    return (z.abs() ** n) * torch.cos(z.angle() * n)
