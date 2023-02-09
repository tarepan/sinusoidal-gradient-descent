"""Experiment runners"""

import math
import os
from typing import Callable, Optional, Sequence, Tuple, Union

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch

from sinusoidal_gradient_descent.core import (
    complex_oscillator,
    estimate_amplitude,
    fft_loss,
    real_oscillator
)
from .metrics import min_lap_cost


class SinusoidEvaluationDataset(torch.utils.data.Dataset):
    """Implements a synthetic dataset of sinusoids in white Gaussian noise for the
    single sinusoid evaluation task. 
    """
    def __init__(
        self,
        signal_length: int = 4096,
        freq_range: Tuple[float] = (0.0, 0.5),
        amp_range: Tuple[float] = (0.0, 1.0),
        phase_range: Tuple[float] = (0.0, 2 * math.pi),
        snr_range: Tuple[float] = (0.0, 30.0),
        n_freqs: int = 100,
        n_amps: int = 100,
        n_phases: int = 100,
        n_snrs: int = 7,
        evaluate_phase: bool = False,
        initial_phase: float = 0.0,
    ):
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
        return self.n_freqs * self.n_amps * self.n_phases * self.n_snrs

    def __getitem__(self, idx):
        if self.evaluate_phase:
            freq_idx = idx % self.n_freqs
            amp_idx = (idx // self.n_freqs) % self.n_amps
            snr_idx = (idx // (self.n_freqs * self.n_amps)) % self.n_snrs
            print(snr_idx)
            phase_idx = (
                idx // (self.n_freqs * self.n_amps * self.n_snrs)
            ) % self.n_phases
        else:
            freq_idx = idx % self.n_freqs
            amp_idx = (idx // self.n_freqs) % self.n_amps
            snr_idx = (idx // (self.n_freqs * self.n_amps)) % self.n_snrs

        freq = (
            (
                self.freq_range[0]
                + (self.freq_range[1] - self.freq_range[0])
                * freq_idx
                / (self.n_freqs - 1)
            )
            if self.n_freqs > 1
            else self.freq_range[0]
        )
        amp = (
            (
                self.amp_range[0]
                + (self.amp_range[1] - self.amp_range[0]) * amp_idx / (self.n_amps - 1)
            )
            if self.n_amps > 1
            else self.amp_range[0]
        )
        phase = (
            (
                self.initial_phase
                + self.phase_range[0]
                + (self.phase_range[1] - self.phase_range[0])
                * phase_idx
                / (self.n_phases - 1)
                if self.n_phases > 1
                else self.phase_range[0]
            )
            if self.evaluate_phase
            else self.initial_phase
        )
        snr = self.snr_range[0] + (self.snr_range[1] - self.snr_range[0]) * snr_idx / (
            self.n_snrs - 1
        ) if self.n_snrs > 1 else self.snr_range[0]
        # noise_stdev = amp / (10 ** (snr / 20))
        snr_linear = 10 ** (snr / 10)
        noise_stdev = amp / ((2 * snr_linear) ** 0.5)

        noise = torch.randn(self.signal_length) * noise_stdev

        n = torch.arange(self.signal_length)
        y = amp * torch.cos(2 * math.pi * freq * n + phase) + noise

        return dict(signal=y, freq=freq, amp=amp, phase=phase, snr=snr, noise_stdev=noise_stdev)


class MultiSinusoidEvaluationDataset(torch.utils.data.Dataset):
    """Implements a synthetic dataset of sinusoidal mixtures for the multi-sinusoid
    estimation task.
    """
    def __init__(
        self,
        signal_length: int = 4096,
        n_components: int = 4,
        freq_range: Tuple[float] = (0.0, 0.5),
        amp_range: Tuple[float] = (0.0, 1.0),
        snr_range: Tuple[float] = (0.0, 30.0),
        n_samples: int = 100,
        n_snrs: int = 7,
        initial_phase: float = 0.0,
        dataset_seed: int = 0,
        enable_noise: bool = True,
    ):
        self.signal_length = signal_length
        self.n_components = n_components
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.snr_range = snr_range
        self.n_samples = n_samples
        self.n_snrs = n_snrs
        self.initial_phase = initial_phase
        self.enable_noise = enable_noise

        with torch.random.fork_rng():
            torch.random.manual_seed(dataset_seed)
            self.freqs = (
                torch.rand(n_samples, n_components) * (freq_range[1] - freq_range[0])
                + freq_range[0]
            )
            self.amps = (
                torch.rand(n_samples, n_components) * (amp_range[1] - amp_range[0])
                + amp_range[0]
            )
            self.amps = self.amps / self.amps.sum(dim=1, keepdim=True) 

    def __len__(self):
        return self.n_samples * self.n_snrs

    def __getitem__(self, idx):
        sample_idx = idx % self.n_samples
        snr_idx = idx // self.n_samples

        freq = self.freqs[sample_idx]
        amp = self.amps[sample_idx]
        snr = (
            self.snr_range[0]
            + (self.snr_range[1] - self.snr_range[0]) * snr_idx / (self.n_snrs - 1)
            if self.n_snrs > 1
            else self.snr_range[0]
        )
        snr_linear = 10 ** (snr / 10)
        noise_stdev = amp.sum() / ((2 * snr_linear) ** 0.5)

        if self.enable_noise:
            noise = torch.randn(self.signal_length) * noise_stdev
        else:
            noise = torch.zeros(self.signal_length)

        n = torch.arange(self.signal_length)
        y = (
            amp[..., None]
            * torch.cos(
                2 * math.pi * freq[..., None] * n[..., None, :] + self.initial_phase
            )
            + noise
        ).sum(dim=0)

        return dict(signal=y, freq=freq, amp=amp, snr=snr, noise_stdev=noise_stdev)


def sample_initial_predictions(
    n_sinusoids: int, # The number of sinusoidal components
    freq_range: Tuple[float], # The range of possible frequencies
    amp_range: Tuple[float], # The range of possible amplitudes
    initial_phase: float, # The initial phase of the sinusoids
    invert_sigmoid: bool = False, # Whether to invert the sigmoid function when sampling the amplitudes
    batch_size: Optional[int] = None, # The batch size of initial predictions
    all_random_in_batch: bool = False, # If true, all predictions in a batch will be sampled randomly. If false, one randomly sampled prediction will be repeated across the batch dimension.
    seed: int = 0, # The random seed
    device: str = "cpu", # The device to place the initial predictions on
    flatten: bool = False, # Whether to flatten the initial predictions
):
    """Samples initial parameters for sinusoidal frequency estimation"""
    shape = (
        (batch_size, n_sinusoids)
        if batch_size is not None and all_random_in_batch
        else (n_sinusoids,)
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        freqs = (
            2
            * math.pi
            * (
                torch.rand(*shape, device=device) * (freq_range[1] - freq_range[0])
                + freq_range[0]
            )
        )
        amps = (
            torch.rand(*shape, device=device) * (amp_range[1] - amp_range[0])
            + amp_range[0]
        )
        phases = torch.ones(*shape, device=device) * initial_phase

        global_amp = torch.ones(*shape, device=device) / n_sinusoids

        if invert_sigmoid:
            # Invert sigmoid so initialisation is in desired range:
            global_amp = torch.log(global_amp / (1 - global_amp))

    if batch_size is not None and not all_random_in_batch:
        freqs = freqs.unsqueeze(0).repeat(batch_size, 1)
        amps = amps.unsqueeze(0).repeat(batch_size, 1)
        phases = phases.unsqueeze(0).repeat(batch_size, 1)
        global_amp = global_amp.unsqueeze(0).repeat(batch_size, 1)

    if flatten:
        freqs = freqs.sum(dim=-1)
        amps = amps.sum(dim=-1)
        phases = phases.sum(dim=-1)
        global_amp = global_amp.sum(dim=-1)

    return freqs, amps, phases, global_amp


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
        normalise_complex_grads - Unused feature (always `False`), but keep alive for future experiments
    """

    saturate_or_id = torch.sigmoid if saturate_global_amp else lambda x:x

    for batch in dataloader:

        # Preparation
        target_signal = batch["signal"].float().to(device)
        target_freq = batch["freq"].float().to(device)
        target_amp = batch["amp"].float().to(device)
        target_snr = batch["snr"].float()

        angle, mag, phase, global_amp = initial_params

        true_batch_size = target_signal.shape[0]
        target_len = target_signal.shape[-1]
        angle = angle[:true_batch_size]
        mag = mag[:true_batch_size]
        phase = phase[:true_batch_size]
        global_amp = global_amp[:true_batch_size]

        if use_real_sinusoid_baseline:
            angle.requires_grad_(True)
            mag.requires_grad_(True)
            optimizer_params = [angle, mag]
        else:
            z = mag * torch.exp(1j * angle)
            z.detach_().requires_grad_(True)
            initial_phase = torch.exp(1j * phase)
            optimizer_params = [z] if not use_global_amp else [z, global_amp]

        optimizer = hydra.utils.instantiate(optimizer_cfg, optimizer_params)
        # /Preparation

        # DiffAbS loop
        for step in range(n_steps):
            # Forward
            optimizer.zero_grad()
            if use_real_sinusoid_baseline:
                pred_signal = real_oscillator(angle, mag, phase, target_len).sum(dim=-2)
            else:
                # Decay sinusoids
                pred_signal = complex_oscillator(
                    z, initial_phase, target_len,
                    # afterwards apply global_amp to each partials | single mode has only 1 partial
                    reduce=False if use_global_amp or mode == "single" else True,
                )
                # Amplitudes
                if use_global_amp:
                    pred_signal = pred_signal * saturate_or_id(global_amp)[..., None]
                    if mode == "multi":
                        # sum all partials
                        pred_signal = pred_signal.sum(dim=-2)
            # /Forward
    
            # Loss-Backward-Optimize
            ## L2 loss (torch.nn.functional.mse_loss) | FT-L2 loss (`core.fft_loss`)
            loss = hydra.utils.call(loss_cfg, pred_signal, target_signal)
            loss.backward()
            # Unused feature, keep alive for future experiments
            if normalise_complex_grads and not use_real_sinusoid_baseline:
                z.grad = z.grad / torch.clamp(z.grad.abs(), min=1e-10)
            optimizer.step()
            # /Loss-Backward-Optimize

            # Logging
            with torch.no_grad():
                if step % log_interval == 0:
                    print(f"Step {step}: {loss.item()}")
                    if not use_real_sinusoid_baseline:
                        if mode == "multi":
                            freq_error = np.mean(min_lap_cost(z.angle().abs() / (2 * math.pi), target_freq.abs(), True))
                        else: # mode == single
                            freq_error = torch.pow(z.angle().abs() / (2 * math.pi) - target_freq.abs(), 2).mean()
                        print(f"Freq error: {freq_error.tolist()}")
            # /Logging

        # /DiffAbS loop

        # Surrogate-to-Sinusoid
        ## Parameter correction
        if use_real_sinusoid_baseline:
            pred_freq, pred_amp = angle, mag
        else:
            # Amplitude correction
            pred_freq = z.angle().abs()
            # `estimate_amplitude` with `representation`
            pred_amp = hydra.utils.call(amplitude_estimator_cfg, z[..., None], target_len)[..., 0]
            # Global amplitude
            if use_global_amp:
                pred_amp = pred_amp * saturate_or_id(global_amp)
        ## Signal generation with corrected parameters
        pred_signal = real_oscillator(pred_freq, pred_amp, phase, target_len).sum(dim=-2)
        # /Surrogate-to-Sinusoid

        # Evaluation: GroundTruth vs fitted Oscillator transferred from Surrogate
        metrics = hydra.utils.call(
            metric_fn_cfg,
            target_signal.detach(), target_freq.detach(),               target_amp.detach(),
            target_snr.detach(),
            pred_signal.detach(),   pred_freq.detach() / (2 * math.pi), pred_amp.detach(),
        )
        # /Evaluation

        metrics["seed"] = seed
        df = pd.DataFrame(metrics)
    return df

# %% ../../nbs/01_evaluate_estimator.ipynb 7
@hydra.main(
    version_base=None, config_path="../../estimator_config", config_name="single"
)
def run(cfg: DictConfig) -> None:
    """Runs the estimator evaluation"""

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = hydra.utils.instantiate(cfg.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True
    )

    amp_estimate_representation = "fft" if "fft" in cfg.loss._target_ else "none"

    metric_df = pd.DataFrame()

    for seed in cfg.seeds:
        initial_params = hydra.utils.call(
            cfg.param_sampler,
            batch_size=cfg.batch_size,
            device=cfg.device,
            seed=seed,
        )

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

        metric_df = pd.concat([metric_df, df_addition], axis=0)

        dir = os.path.split(cfg.output_file)[0]
        if not os.path.exists(dir):
            os.makedirs(dir)
        metric_df.to_csv(cfg.output_file, index=False)
