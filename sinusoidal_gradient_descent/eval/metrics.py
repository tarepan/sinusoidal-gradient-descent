"""Evaluation metrics"""

from lapsolver import solve_dense
import numpy as np
import torch


def _magnitude_spectrum(x):
    return torch.abs(torch.fft.rfft(x, norm="forward"))


def spectral_mse(x, y):
    return torch.mean((_magnitude_spectrum(x) - _magnitude_spectrum(y)) ** 2)


def min_lap_cost(target, predicted, unsqueeze=False):
    """Calculate the minimum lap cost between two sets of points."""
    if unsqueeze:
        target = target[..., None]
        predicted = predicted[..., None]

    cost = (torch.cdist(target, predicted) ** 2).cpu().numpy()
    costs = []
    for i in range(cost.shape[0]):
        r, c = solve_dense(cost[i])
        costs.append(cost[i, r, c].mean())
    return costs


def single_sinusoid_metrics(
    target_signal: torch.Tensor,
    target_freq: torch.Tensor,
    target_amp: torch.Tensor,
    target_snr: torch.Tensor,
    predicted_signal: torch.Tensor,
    predicted_freq: torch.Tensor,
    predicted_amp: torch.Tensor,
):
    """Calculate metrics for single sinusoid evaluation.
    
    metrics: MSE of amplitude, frequency, joint, signal [linear|dB]
    """

    data = dict(
        target_freq=target_freq.cpu().numpy(),
        target_amp=target_amp.cpu().numpy(),
        target_snr=target_snr.cpu().numpy(),
        predicted_freq=predicted_freq.cpu().numpy(),
        predicted_amp=predicted_amp.cpu().numpy(),
    )

    # MSE
    ## frequency, amplitude and signal
    data["freq_mse"] = ((target_freq - predicted_freq) ** 2).cpu().numpy()
    data["amp_mse"]  = ((target_amp  - predicted_amp)  ** 2).cpu().numpy()
    data["signal_mse"] = torch.mean((target_signal - predicted_signal) ** 2, dim=-1).cpu().numpy()
    ## Joint
    joint_target    = torch.stack((target_freq,    target_amp),    dim=-1)
    joint_predicted = torch.stack((predicted_freq, predicted_amp), dim=-1)
    data["joint_mse"] = torch.mean((joint_target - joint_predicted) ** 2, dim=-1).cpu().numpy()

    # dB
    data["freq_mse_db"]   = 10 * np.log10(data["freq_mse"])
    data["amp_mse_db"]    = 10 * np.log10(data["amp_mse"])
    data["signal_mse_db"] = 10 * np.log10(data["signal_mse"])
    data["joint_mse_db"]  = 10 * np.log10(data["joint_mse"])

    return data


def multi_sinusoid_metrics(
    target_signal: torch.Tensor,    target_freq: torch.Tensor,    target_amp: torch.Tensor,    target_snr: torch.Tensor,
    predicted_signal: torch.Tensor, predicted_freq: torch.Tensor, predicted_amp: torch.Tensor,
):
    """Calculate metrics for single sinusoid evaluation."""

    # 'linear assignment cost' or 'Chamfer distance' are removed because of poor correlation with perceptual similarity

    data = dict(
        target_freq=[t.numpy() for t in target_freq.cpu().split(1)],
        target_amp=[t.numpy() for t in target_amp.cpu().split(1)],
        target_snr=target_snr.cpu().numpy(),
        predicted_freq=[t.numpy() for t in predicted_freq.cpu().split(1)],
        predicted_amp=[t.numpy() for t in predicted_amp.cpu().split(1)],
    )

    # Signal MSE
    data["signal_mse"] = torch.mean((target_signal - predicted_signal) ** 2, dim=-1).cpu().numpy()
    data["signal_mse_db"] = 10 * np.log10(data["signal_mse"])
    # Spectral MSE
    data["spectral_mse"] = 10 * np.log10(
        np.stack(
            list(map(spectral_mse, target_signal.cpu().split(1), predicted_signal.cpu().split(1))),
            axis=0,
        )
    )

    return data
