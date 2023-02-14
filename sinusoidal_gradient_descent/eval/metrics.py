"""Evaluation metrics"""

from lapsolver import solve_dense
import numpy as np
import torch
import torch.nn.functional as F


def _magnitude_spectrum(x):
    return torch.abs(torch.fft.rfft(x, norm="forward"))


def spectral_mse(x, y):
    """Calculate spectral MSE between input's spectrums.

    Returns:
        mse :: () - MSE as PyTorch scalar
    """
    return torch.nn.functional.mse_loss(_magnitude_spectrum(x), _magnitude_spectrum(y))


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
    target_signal: torch.Tensor,       target_freq: torch.Tensor,    target_amp: torch.Tensor, target_snr: torch.Tensor,
    predicted_signal: torch.Tensor, predicted_freq: torch.Tensor, predicted_amp: torch.Tensor,
):
    """Calculate metrics for single-sinusoid evaluation.
    
    There is only one sinusoid in a signal, so loss of frequency and amplitude can be evaluated one-by-one.
    We evaluate frequency / ampliutde / signal / freq-amp-joint MSE.

    Args:
        target_signal
        target_freq
        target_amp
        target_snr
        predicted_signal :: (B, L) - Predicted signals
        predicted_freq   :: (B,)   - Predicted frequencies
        predicted_amp    :: (B,)   - Predicted amplitudes
    Returns:
        Dict
            target_freq
            target_amp
            target_snr
            predicted_freq :: (B,) - Predicted frequencies [?rad?]
            predicted_amp  :: (B,) - Predicted amplitudes
            freq_mse       :: (B,) - Losses, frequency MSE
            freq_mse_db    :: (B,) - Losses, frequency MSE [dB]
            amp_mse        :: (B,) - Losses, amplitude MSE
            amp_mse_db     :: (B,) - Losses, amplitude MSE [dB]
            signal_mse     :: (B,) - Losses, singal    MSE
            signal_mse_db  :: (B,) - Losses, singal    MSE [dB]
            joint_mse      :: (B,) - Losses, joint     MSE
            joint_mse_db   :: (B,) - Losses, joint     MSE [dB]
    """

    data = dict(
        target_freq    = target_freq.cpu().numpy(),
        target_amp     = target_amp.cpu().numpy(),
        target_snr     = target_snr.cpu().numpy(),
        predicted_freq = predicted_freq.cpu().numpy(),
        predicted_amp  = predicted_amp.cpu().numpy(),
    )

    ## Joint (TODO: I don't understand meaning of joint yet)
    target_joint    = torch.stack((target_freq,    target_amp),    dim=-1)
    predicted_joint = torch.stack((predicted_freq, predicted_amp), dim=-1)

    # MSE
    ## frequency :: (B,) -> (B,) / amplitude :: (B,) -> (B,) / signal :: (B, L) -> (B,) / joint :: ? (2B,) -> (2B,)
    data["freq_mse"]   =           ((target_freq   - predicted_freq  ) ** 2).cpu().numpy()
    data["amp_mse"]    =           ((target_amp    - predicted_amp   ) ** 2).cpu().numpy()
    data["signal_mse"] = torch.mean((target_signal - predicted_signal) ** 2, dim=-1).cpu().numpy()
    data["joint_mse"]  = torch.mean((target_joint  - predicted_joint ) ** 2, dim=-1).cpu().numpy()

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
    """Calculate metrics for multi-sinusoids evaluation.

    There are multiple sinusoids in a signal, so loss of frequency and amplitude can not be evaluated one-by-one manner.
    We evaluate time-domain signal MSE and frequency-domain spectral MSE.

    Args:
        target_signal
        target_freq
        target_amp
        target_snr
        predicted_signal :: (B, L) - Predicted signals
        predicted_freq   :: (B, K) - Predicted frequencies
        predicted_amp    :: (B, K) - Predicted amplitudes
    Returns:
        Dict
            target_freq
            target_amp
            target_snr
            predicted_freq  :: (B, K) - Predicted frequencies [?rad?]
            predicted_amp   :: (B, K) - Predicted amplitudes
            signal_mse      :: (B,)   - Losses, singal   MSE
            signal_mse_db   :: (B,)   - Losses, singal   MSE [dB]
            spectral_mse    :: (B,)   - Losses, spectral MSE
            spectral_mse_db :: (B,)   - Losses, spectral MSE [dB]
    """

    # 'linear assignment cost' or 'Chamfer distance' are removed because of poor correlation with perceptual similarity

    data = dict(
        target_freq    = [t.numpy() for t in target_freq.cpu().split(1)],
        target_amp     = [t.numpy() for t in target_amp.cpu().split(1)],
        target_snr     = target_snr.cpu().numpy(),
        predicted_freq = [t.numpy() for t in predicted_freq.cpu().split(1)],
        predicted_amp  = [t.numpy() for t in predicted_amp.cpu().split(1)],
    )

    # MSE
    ## Signal MSE :: (B, L) -> (B,)
    data["signal_mse"] = torch.mean((target_signal - predicted_signal) ** 2, dim=-1).cpu().numpy()
    ## Spectral MSE :: (B, L) -> List[(L,)] -> List[()] -> (B,)
    data["spectral_mse"] = np.stack(list(map(spectral_mse, target_signal.cpu().split(1), predicted_signal.cpu().split(1))), axis=0)

    # dB
    data["signal_mse_db"]   = 10 * np.log10(data["signal_mse"])
    data["spectral_mse_db"] = 10 * np.log10(data["spectral_mse"])

    return data
