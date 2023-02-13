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


def complex_oscillator(z: torch.ComplexType, initial_phase: Optional[torch.ComplexType] = None, N: int = 2048):
    """Generates exponentially decaying sinusoids from representative z.

    Args:
        z :: (...) - Time-invariant complex numbers representing the decaying sinusoids
        initial_phase :: (...) - Initial phases as complex numbers on unit circle
        N - Length of output sinusoid
    Returns:
        decaying_sinusoids :: (..., N) - Exponentially decaying sinusoids
    """

    # φ = initial_phase ? 0
    initial_phase = initial_phase if (initial_phase is not None) else torch.ones_like(z)

    # cumulative product :: (...) -> (..., N) - Convert z and φ to [φ, z, z, ..., z] to [φ, φ*z, φ*z*z, ..., φ*z^(N-1)]
    z_series = torch.cat([initial_phase.unsqueeze(-1), z.unsqueeze(-1).expand(*z.shape, N-1)], dim=-1)
    decaying_sinusoids = z_series.cumprod(dim=-1).real

    return decaying_sinusoids


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
