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


def get_reduce_fn(reduce: str):
    if reduce == "mean":
        return torch.mean
    elif reduce == "sum":
        return torch.sum
    elif reduce == "none":
        return lambda x: x
    else:
        raise ValueError(f"Invalid reduction method: {reduce}")


def fft_loss(
    pred_signal,
    target_signal,
    lin_l1: float = 1.0,
    lin_l2: float = 0.0,
    lin_huber: float = 0.0,
    log_l1: float = 0.0,
    log_l2: float = 0.0,
    log_huber: float = 0.0,
    reduce_freq: str = "mean",
    reduce_batch: str = "sum",
    eps: float = 1e-8,
):
    """
    Args:
        reduce_freq  - Identifier of reduce function for freq
        reduce_batch - Identifier of reduce function for batch
    """

    freq_reduce_fn = get_reduce_fn(reduce_freq)
    batch_reduce_fn = get_reduce_fn(reduce_batch)

    pred_fft = torch.fft.rfft(pred_signal, norm="ortho").abs()
    target_fft = torch.fft.rfft(target_signal, norm="ortho").abs()
    lin_l1 = (
        lin_l1 * torch.nn.functional.l1_loss(pred_fft, target_fft)
        if lin_l1 != 0.0
        else 0.0
    )
    lin_l2 = (
        lin_l2 * torch.nn.functional.mse_loss(pred_fft, target_fft)
        if lin_l2 != 0.0
        else 0.0
    )
    lin_huber = (
        lin_huber * torch.nn.functional.huber_loss(pred_fft, target_fft)
        if lin_huber != 0.0
        else 0.0
    )
    log_l1 = (
        log_l1
        * torch.nn.functional.l1_loss(
            torch.log(pred_fft + eps), torch.log(target_fft + eps)
        )
        if log_l1 != 0.0
        else 0.0
    )
    log_l2 = (
        log_l2
        * torch.nn.functional.mse_loss(
            torch.log(pred_fft + eps), torch.log(target_fft + eps)
        )
        if log_l2 != 0.0
        else 0.0
    )
    log_huber = (
        log_huber
        * torch.nn.functional.huber_loss(
            torch.log(pred_fft + eps), torch.log(target_fft + eps)
        )
        if log_huber != 0.0
        else 0.0
    )

    return lin_l1 + lin_l2 + lin_huber + log_l1 + log_l2 + log_huber
