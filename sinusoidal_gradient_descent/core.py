"""Core of sinusoidal gradient descent"""

import math
from typing import Optional

import torch


def real_oscillator(freqs, amps, phases=None, N=2048):
    """Generate real sinusoids."""
    freqs = freqs[..., None]
    amps = amps[..., None]
    phases = phases[..., None] if phases is not None else torch.zeros_like(freqs)
    n = torch.arange(N, dtype=torch.float32, device=freqs.device)[..., None, :]
    return torch.cos(freqs * n + phases) * amps


def complex_oscillator(
    z: torch.ComplexType,
    initial_phase: Optional[torch.ComplexType] = None,
    N: int = 2048,
    reduce: bool = False,
):
    """Generates an exponentially decaying sinusoids."""

    if initial_phase is None:
        # If no provided, initialized with zero phase.
        initial_phase = torch.ones_like(z)

    # 'cumulative product' implementation of the surrogate
    z = z[..., None].expand(*z.shape, N - 1)
    z = torch.cat([initial_phase.unsqueeze(-1), z], dim=-1)
    y = z.cumprod(dim=-1).real

    if reduce:
        y = y.sum(dim=-2)

    return y


def estimate_amplitude(z, N, representation="fft"):
    """Estimate amplitude of real oscillator from surrogate complex oscillator.

    Args:
        z - complex numbers representing complex oscillator
        N - The number of samples
        representation - Estimation based on FFT-MSE or normal MSE
    """

    # Source signal
    V = complex_oscillator(z, N=N).sum(dim=-2)

    # Target real oscillator with amplutide 1
    n = torch.arange(N, device=z.device)
    omega = torch.angle(z)
    U = torch.cos(omega[..., None, :] * n[None, :, None])

    # Amplitude estimation
    ## FFT-MSE or MSE
    if representation == "fft":
        U = torch.fft.rfft(U, dim=-2).abs()
        V = torch.fft.rfft(V).abs()
    ## least-square estimation
    least_squares_soln = torch.linalg.lstsq(U, V).solution

    return least_squares_soln


def fft_loss(pred_signal, target_signal):
    """
    L2 Loss between FT(pred_signal) and FT(target_signal).
    """
    # original code: weight sum of six [linear|log]x[L1|L2|Huber] losses
    return torch.nn.functional.mse_loss(
        torch.fft.rfft(pred_signal,   norm="ortho").abs(),
        torch.fft.rfft(target_signal, norm="ortho").abs()
    )
