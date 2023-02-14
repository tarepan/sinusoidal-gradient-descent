"""Experiment runners"""

import math
import os
from typing import Callable, Optional, Sequence, Tuple, Union

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sinusoidal_gradient_descent.core import (
    complex_oscillator,
    estimate_amplitude,
    fft_loss,
    real_oscillator
)
from .metrics import min_lap_cost


def sample_equally_devided(low_high: Tuple[float], sample_idx: int, n_division: int) -> float:
    """Sample a value from a equally devided range with index.

    If `lowest != highest` but `n_division == 1`, use lowest value as a output.

    Args:
    low_high - Lowest and Highest value of the sampling range
    sample_idx - Sampling index starting from 0
    n_division - The number of range division
    """

    if n_division == 1:
        return low_high[0]
    else:
        return low_high[0] + (low_high[1] - low_high[0]) * sample_idx / (n_division - 1)


def test_sample_equally_devided():
    assert sample_equally_devided([0.0, 1.0], 1, 3) == 0.5
    assert sample_equally_devided([0.0, 1.0], 1, 1) == 0.0
    print("done 'test_sample_equally_devided'")


def sample_from_range(low_high: Tuple[float], shape: Tuple[int]) -> torch.Tensor:
    """Sample a random tensor from within the range.

    Args:
        low_high - Lowest and Highest value of the sampling range
        shape - Sampled tensor shape
    """
    return torch.rand(*shape) * (low_high[1] - low_high[0]) + low_high[0]


class SinusoidEvaluationDataset(torch.utils.data.Dataset):
    """Single sinusoids."""

    def __init__(
        self,
        signal_length: int,
        n_components: int,
        freq_range:  Tuple[float] = (0.0,  0.5),
        amp_range:   Tuple[float] = (0.0,  1.0),
        phase_range: Tuple[float] = (0.0,  2 * math.pi),
        snr_range:   Tuple[float] = (0.0, 30.0),
        n_samples: int = 100,
        n_freqs:  int = 100,
        n_amps:   int = 100,
        n_phases: int = 100,
        n_snrs: int = 7,
        evaluate_phase: bool = False,
        initial_phase: float = 0.0,
        dataset_seed: int = 0,
        enable_noise: bool = True,
    ):
        """
        Args:
            signal_length - Length of a signal
            n_components - The |K|, the number of sinusoidal components in a signal
            freq_range  - Lowest/Highest value of frequency
            amp_range   - Lowest/Highest value of amplitude
            phase_range - Lowest/Highest value of phase
            snr_range   - Lowest/Highest value of snr
            n_samples - The number of samples (signals)           in a dataset
            n_freqs   - The number of equally-devided frequencies in a dataset
            n_amps    - The number of equally-devided amplitudes  in a dataset
            n_phases  - The number of equally-devided phases      in a dataset
            n_snrs    - The number of equally-devided noise level toward a signal
            evaluate_phase
            initial_phase
            dataset_seed - Random sampling seed
            enable_noise - Whether to add white noise
        """

        assert n_components == 1, "Single dataset should be K==1."

        # Just save as fields
        self.signal_length = signal_length
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.snr_range = snr_range
        self.n_freqs = n_freqs
        self.n_amps = n_amps
        self.n_snrs = n_snrs
        self.initial_phase = initial_phase
        self.evaluate_phase = evaluate_phase
        if self.evaluate_phase:
            self.phase_range = phase_range
            self.n_phases = n_phases
        else:
            self.phase_range = (0, 0)
            self.n_phases = 1

    def __len__(self):
        # 'frequency variation' * 'amplitude variation' * 'phase variation' * 'noise variation'
        return self.n_freqs * self.n_amps * self.n_phases * self.n_snrs

    def __getitem__(self, idx):
        """
        Returns:
            Dict
                signal -
                freq -
                amp -
                phase -
                snr -
                noise_stdev -
        """

        # Indexes
        freq_idx = idx % self.n_freqs
        amp_idx = (idx // self.n_freqs) % self.n_amps
        snr_idx = (idx // (self.n_freqs * self.n_amps)) % self.n_snrs
        if self.evaluate_phase:
            print(snr_idx)
            phase_idx = (idx // (self.n_freqs * self.n_amps * self.n_snrs)) % self.n_phases

        # Parameter sampling
        freq = sample_equally_devided(self.freq_range, freq_idx, self.n_freqs)
        amp  = sample_equally_devided(self.amp_range,  amp_idx,  self.n_amps)
        phase = self.initial_phase + (sample_equally_devided(self.phase_range, phase_idx, self.n_phases) if self.evaluate_phase else 0.0)
        ## White noise
        snr = sample_equally_devided(self.snr_range, snr_idx, self.n_snrs)
        snr_linear = 10 ** (snr / 10)
        noise_stdev = amp / ((2 * snr_linear) ** 0.5)
        white_noise = torch.randn(self.signal_length) * noise_stdev

        # Signal generation :: (1,) * (L=N,) -> (L=N,)
        n = torch.arange(self.signal_length)
        y = white_noise + amp * torch.cos(2 * math.pi * freq * n + phase)

        return dict(signal=y, freq=freq, amp=amp, phase=phase, snr=snr, noise_stdev=noise_stdev)


class MultiSinusoidEvaluationDataset(torch.utils.data.Dataset):
    """Mixtures of sinusoids."""
    def __init__(
        self,
        signal_length: int,
        n_components: int,
        freq_range:  Tuple[float] = (0.0,  0.5),
        amp_range:   Tuple[float] = (0.0,  1.0),
        phase_range: Tuple[float] = (0.0,  2 * math.pi),
        snr_range:   Tuple[float] = (0.0, 30.0),
        n_samples: int = 100,
        n_freqs:  int = 100,
        n_amps:   int = 100,
        n_phases: int = 100,
        n_snrs: int = 7,
        evaluate_phase: bool = False,
        initial_phase: float = 0.0,
        dataset_seed: int = 0,
        enable_noise: bool = True,
    ):
        """
        Args:
            signal_length - Length of a signal
            n_components - The |K|, the number of sinusoidal components in a signal
            freq_range  - Lowest/Highest value of frequency
            amp_range   - Lowest/Highest value of amplitude
            phase_range - Lowest/Highest value of phase
            snr_range   - Lowest/Highest value of snr
            n_samples - The number of clean signals (w/o noise)   in a dataset
            n_freqs   - The number of equally-devided frequencies in a dataset
            n_amps    - The number of equally-devided amplitudes  in a dataset
            n_phases  - The number of equally-devided phases      in a dataset
            n_snrs    - The number of equally-devided noise level toward a signal
            evaluate_phase
            initial_phase
            dataset_seed - Random sampling seed
            enable_noise - Whether to add white noise
        """

        self.signal_length = signal_length
        self.snr_range = snr_range
        self.n_samples = n_samples
        self.n_snrs = n_snrs
        self.initial_phase = initial_phase
        self.enable_noise = enable_noise

        # Sampling
        with torch.random.fork_rng():
            torch.random.manual_seed(dataset_seed)
            shape = (n_samples, n_components)
            ## freqs :: Tensor[N, K] - ~ U[freq_range]
            self.freqs =  sample_from_range(freq_range, shape)
            ## amps :: Tensor[N, K] - ~ U[amp_range], then normalized to total 1 in each signals
            unnorm_amps = sample_from_range(amp_range,  shape)
            self.amps = unnorm_amps / unnorm_amps.sum(dim=1, keepdim=True) 

    def __len__(self):
        # 'clean signals' * 'variation of noise level'
        return self.n_samples * self.n_snrs

    def __getitem__(self, idx):
        """
        returns miss `phase` compared to `SinusoidEvaluationDataset`.

        Returns:
            Dict
                signal -
                freq -
                amp -
                snr -
                noise_stdev -
        """

        # Indexes
        sample_idx = idx %  self.n_samples
        snr_idx    = idx // self.n_samples

        # Query
        ## freq :: Tensor[K]
        freq = self.freqs[sample_idx]
        ## amp  :: Tensor[K]
        amp =   self.amps[sample_idx]

        # Parameter sampling
        ## White noise or Silent
        snr = sample_equally_devided(self.snr_range, snr_idx, self.n_snrs)
        snr_linear = 10 ** (snr / 10)
        noise_stdev = amp.sum() / ((2 * snr_linear) ** 0.5)
        noise_or_silent = torch.randn(self.signal_length) * noise_stdev if self.enable_noise else torch.zeros(self.signal_length)

        # Signal generation :: (K, L=1) * (L=N,) -> (K, L=N) -> (L=N,)
        n = torch.arange(self.signal_length)
        ys = noise_or_silent + amp.unsqueeze(-1) * torch.cos(2 * math.pi * freq.unsqueeze(-1) * n + self.initial_phase)
        y = ys.sum(dim=0)

        return dict(signal=y, freq=freq, amp=amp, snr=snr, noise_stdev=noise_stdev)


def fill_batch(item, batch_size):
    """Fill a batch with copy of the item."""
    # (*) -> (1, *) -> (B, *)
    return item.unsqueeze(0).repeat(batch_size, 1)


def sample_initial_predictions(
    n_sinusoids: int,                  # The number of sinusoidal components
    freq_range: Tuple[float],          # The range of possible frequencies
    amp_range: Tuple[float],           # The range of possible amplitudes
    initial_phase: float,              # The initial phase of the sinusoids
    invert_sigmoid: bool = False,      # Whether to use `global_amp` as `logit(a_k)`, enabling [0, 1]-bounded a_k with `sigmoid(global_amp)`
    batch_size: int = 0,               # The batch size of initial predictions
    all_random_in_batch: bool = False, # If true, all predictions in a batch will be sampled randomly. If false, one randomly sampled prediction will be repeated across the batch dimension.
    seed: int = 0,                     # The random seed
    device: str = "cpu",               # The device to place the initial predictions on
    use_real_sinusoid: bool = False,
):
    """Samples initial parameters for sinusoidal frequency estimation.

    Outputs:
        spin      :: (B, K) - Random sampling from within the range [rad] | z from random-sampled freq and magnitude
        amplitude :: (B, K) - Random sampling from within the range       | 1/|K|, could be logit form
        phase     :: (B, K) - Based on specifier argument
    """

    use_r = use_real_sinusoid

    # Shape of 4 params : (B, K) | (K,)
    shape = (batch_size, n_sinusoids) if all_random_in_batch else (n_sinusoids,)

    # Sampling
    with torch.random.fork_rng():
        torch.manual_seed(seed)

        freq       = 2 * math.pi * sample_from_range(freq_range, shape).to(device)
        mag_linear =               sample_from_range(amp_range,  shape).to(device)
        phase_rad  = torch.ones(*shape, device=device) * initial_phase
        global_amp = torch.ones(*shape, device=device) / n_sinusoids

        # Parameters : (real | complex) - (freq | z), (phase_rad | complex phase component), (mag_linear | global_amp)
        spin       = freq       if use_r else (mag_linear * torch.exp(1j * freq)).detach()
        phase      = phase_rad  if use_r else torch.exp(1j * phase_rad)
        amplitude  = mag_linear if use_r else global_amp
        if invert_sigmoid:
            amplitude = torch.special.logit(amplitude)

    if not all_random_in_batch:
        # (K,) -> (B, K)
        spin, amplitude, phase = fill_batch(spin, batch_size) ,fill_batch(amplitude, batch_size), fill_batch(phase, batch_size)

    return spin, amplitude, phase


def abs_freq(z):
    """Extract absolute frequency (not angular frequency) from comlex number."""
    return z.angle().abs() / (2 * math.pi)


def pick_activated_items(batch, names, device):
    """Pick items from a batch with activation as Tensor."""
    return list(map(lambda name: batch[name].float().to(device), names))


def evaluation_loop(
    dataloader: torch.utils.data.DataLoader,
    loss_cfg: DictConfig,
    optimizer_cfg: DictConfig,
    amplitude_estimator_cfg: DictConfig,
    metric_fn_cfg: DictConfig,
    initial_params: Tuple[torch.Tensor],
    use_real_sinusoid_baseline: bool = False,
    use_global_amp: bool = True,
    saturate_global_amp: bool = False,
    normalise_complex_grads: bool = False,
    mode: str = "multi",
    device: Union[torch.device, str] = "cpu",
    n_steps: int = 1000,
    log_interval: int = 100,
    seed: int = 0,
):
    """Runs the experimental evaluation
    
    Args:
        saturate_global_amp - Whether to use `global_amp` as `logit(a_k)`, enabling [0, 1]-bounded a_k with `sigmoid(global_amp)`
        normalise_complex_grads - Unused feature (always `False`), but keep alive for future experiments
    """

    use_r = use_real_sinusoid_baseline

    # Purpose: Keep in [0, 1]
    # This is not general way, but it is correct for the paper's experiment because a reference signal's amplitude α_k is always <1.
    saturate_or_id = torch.sigmoid if saturate_global_amp else lambda x:x

    for batch in dataloader:

        # Preparation
        ## Target ::
        target_signal, target_freq, target_amp, target_snr = pick_activated_items(["signal", "freq", "amp", "snr"])
        batch_size, length = target_signal.shape[0], target_signal.shape[-1]
        ## Parameters :: (B', K) -> (B, K)
        spin, amplitude, phase = initial_params
        spin, amplitude, phase = spin[:batch_size], amplitude[:batch_size], phase[:batch_size]

        ## Model
        oscillator = real_oscillator if use_r else complex_oscillator

        ## Optimizer
        ### spin
        optimizer_params = []
        spin.requires_grad_(True)
        optimizer_params += [spin]
        ### amplitude
        if use_global_amp:
            amplitude.requires_grad_(True)
            optimizer_params += [amplitude]
        ### phase
        # phase.requires_grad_(True)
        # optimizer_params += [phase]
        ### optim
        optimizer = hydra.utils.instantiate(optimizer_cfg, optimizer_params)

        # /Preparation

        # DiffAbS loop
        for step in range(n_steps):
            # Forward :: (B, K) -> (B, K, L) -> (B, L)
            optimizer.zero_grad()
            pred_signal = oscillator(spin, phase, length)
            if use_global_amp:
                pred_signal = saturate_or_id(amplitude).unsqueeze(-1) * pred_signal
            pred_signal = pred_signal.sum(dim=-2)

            # Loss-Backward-Optimize
            loss = hydra.utils.call(loss_cfg, pred_signal, target_signal)
            loss.backward()
            optimizer.step()

            # Logging
            with torch.no_grad():
                if step % log_interval == 0:
                    print(f"Step {step}: {loss.item()}")
                    if not use_r:
                        # MinLapCost for multi, MSE for single
                        if mode == "multi":
                            freq_error = np.mean(min_lap_cost(abs_freq(spin), target_freq.abs(), True))
                        else:
                            freq_error = F.mse_loss(abs_freq(spin), target_freq.abs())
                        print(f"Freq error: {freq_error.tolist()}")

        # Evaluation: GroundTruth vs fitted Oscillator transferred from Surrogate

        ## Surrogate-to-Sinusoid
        ### Parameter correction, pred_freq :: (B, K), pred_amp :: (B, K)
        if use_r:
            pred_freq, pred_amp = spin, amplitude
        else:
            pred_freq = spin.angle().abs()
            # Amplitude correction
            # TODO: Check whether removed `flatten` cause bug here or not
            pred_amp = hydra.utils.call(amplitude_estimator_cfg, spin.unsqueeze(-1), length)[..., 0]
            if use_global_amp:
                pred_amp = pred_amp * saturate_or_id(amplitude)
        ### Signal generation :: (B, K) -> (B, K, L) -> (B, L)
        pred_signal = (pred_amp.unsqueeze(-1) * real_oscillator(pred_freq, phase_rad, length)).sum(dim=-2)

        ## Metrics
        if mode is not "multi":
            # (B, K=1) -> (B,)
            pred_freq, pred_amp = pred_freq.unsqueeze(-1), pred_amp.unsqueeze(-1)
        metrics = hydra.utils.call(
            metric_fn_cfg,
            target_signal.detach(), target_freq.detach(),               target_amp.detach(), target_snr.detach(),
            pred_signal.detach(),   pred_freq.detach() / (2 * math.pi), pred_amp.detach(),
        )
        metrics["seed"] = seed
        df = pd.DataFrame(metrics)

    return df


@hydra.main(version_base=None, config_path="../../estimator_config", config_name="single")
def run(cfg: DictConfig) -> None:
    """Runs the estimator evaluation"""

    # Env Config
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataset/Loader
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Output container
    metric_df = pd.DataFrame()

    for seed in cfg.seeds:

        # Instantiate initial parameters
        initial_params = hydra.utils.call(cfg.param_sampler, batch_size=cfg.batch_size, device=cfg.device, seed=seed)

        # Run
        df_addition = hydra.utils.call(
            cfg.evaluation,
            dataloader,
            cfg.loss,
            cfg.optimizer,
            cfg.amplitude_estimator,
            cfg.metric_fn,
            initial_params,
            device=cfg.device,
            n_steps=cfg.n_steps,
            log_interval=cfg.log_interval,
            seed=seed,
        )

        # Save in container
        metric_df = pd.concat([metric_df, df_addition], axis=0)

        # Write
        out_dir = os.path.split(cfg.output_file)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        metric_df.to_csv(cfg.output_file, index=False)

if __name__ == "__main__":
    # use hydra
    run()