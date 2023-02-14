"""Core of sinusoidal gradient descent"""

import math
from typing import Optional

import torch


def real_oscillator(freq, phase = None, length: int = 2048):
    """Generates real sinusoids.
    
    Args:
        freq  :: (*) - Time-invariant angular frequencies [rad/sample]
        phase :: (*) - Initial phase [rad]
        length - Length of output sinusoid
    Returns:
        sinusoids :: (*, L=N) - a * cos(ωn + φ)
    """

    ## Init : φ = phase ? 0
    phase = phase if (phase is not None) else torch.zeros_like(freq)

    # Reshape :: (*) -> (*, L=1)
    freq, phase = freq.unsqueeze(-1), phase.unsqueeze(-1)

    # Direct sinusoids :: (*, L=1) * (L=L,) -> (*, L=L)
    n = torch.arange(length, dtype=torch.float32, device=freq.device)
    return torch.cos(freq * n + phase)


def complex_oscillator(z: torch.ComplexType, phase: Optional[torch.ComplexType] = None, length: int = 2048):
    """Generates exponentially decaying sinusoids.

    Args:
        z :: (*) - Time-invariant complex numbers representing the decaying sinusoids
        phase :: (*) - Initial phase components as complex numbers on unit circle (not argument)
        length - Length of output sinusoid
    Returns:
        decaying_sinusoids :: (*, L=L) - Exponentially decaying sinusoids, Re(phase * z^n) = |z|^n * cos(n∠z + ∠phase)
    """

    # Init : φ = phase ? e^0j (==1)
    phase = phase if (phase is not None) else torch.ones_like(z)

    # Reshape :: (*) -> (*, L=1)
    shape = z.shape
    z, phase = z.unsqueeze(-1), phase.unsqueeze(-1)

    # Cumulative product :: (*, L=1) -> (*, L=L) - Convert z and φ to [φ, z, z, ..., z] to [φ, φ*z, φ*z*z, ..., φ*z^(L-1)]
    return torch.cat([phase, z.expand(*shape, length-1)], dim=-1).cumprod(dim=-1).real


def estimate_amplitude(z, length: int, representation="fft"):
    """Estimate amplitude of real oscillator from surrogate complex oscillator.

    Args:
        z :: (*, 1)? - complex numbers representing complex oscillator
        length - The number of samples
        representation - Estimation based on FFT-MSE or normal MSE
    """

    # Source signal :: (*, 1) -> (*, 1, L) -> (*, L)
    V = complex_oscillator(z, length=length).sum(dim=-2)

    # Target real oscillator with amplutide 1
    n = torch.arange(length, device=z.device)
    ## (*, 1)
    freq = torch.angle(z)
    ## (*, 1, 1) * (1, L, 1) -> maybe (*, L, 1)
    U = torch.cos(freq[..., None, :] * n[None, :, None])

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
