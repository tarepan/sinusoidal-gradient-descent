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


def sample_equally_devided(low_high: Tuple[float], sample_idx: int, n_division: int):
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

        # Signal generation
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

        # Signal generation
        n = torch.arange(self.signal_length)
        ys = noise_or_silent + amp[..., None] * torch.cos(2 * math.pi * freq[..., None] * n[..., None, :] + self.initial_phase)
        y = ys.sum(dim=0)

        return dict(signal=y, freq=freq, amp=amp, snr=snr, noise_stdev=noise_stdev)


def sample_initial_predictions(
    n_sinusoids: int,                  # The number of sinusoidal components
    freq_range: Tuple[float],          # The range of possible frequencies
    amp_range: Tuple[float],           # The range of possible amplitudes
    initial_phase: float,              # The initial phase of the sinusoids
    invert_sigmoid: bool = False,      # Whether to use `global_amp` as `logit(a_k)`, enabling [0, 1]-bounded a_k with `sigmoid(global_amp)`
    batch_size: Optional[int] = None,  # The batch size of initial predictions
    all_random_in_batch: bool = False, # If true, all predictions in a batch will be sampled randomly. If false, one randomly sampled prediction will be repeated across the batch dimension.
    seed: int = 0,                     # The random seed
    device: str = "cpu",               # The device to place the initial predictions on
    flatten: bool = False,             # Whether to flatten the initial predictions
):
    """Samples initial parameters for sinusoidal frequency estimation.

    Outputs:
        freqs - Random sampling from within the range
        amps - Random sampling from within the range
        phases - Based on specifier argument
        global_amp - 1/|K|, equal between components and sum to 1
    """

    # Shape of 4 params : (n_batch, n_components) | (n_components,)
    shape = (batch_size, n_sinusoids) if (batch_size is not None and all_random_in_batch) else (n_sinusoids,)

    # Sampling
    with torch.random.fork_rng():
        # Rules are described above.
        torch.manual_seed(seed)
        freqs = 2 * math.pi * sample_from_range(freq_range, shape).to(device)
        amps = sample_from_range(amp_range, shape).to(device)
        phases = torch.ones(*shape, device=device) * initial_phase
        global_amp = torch.ones(*shape, device=device) / n_sinusoids
        if invert_sigmoid:
            global_amp = torch.special.logit(global_amp)

    if batch_size is not None and not all_random_in_batch:
        # Tensor[K,] -> Tensor[1, K] -> Tensor[N, K]
        freqs      =      freqs.unsqueeze(0).repeat(batch_size, 1)
        amps       =       amps.unsqueeze(0).repeat(batch_size, 1)
        phases     =     phases.unsqueeze(0).repeat(batch_size, 1)
        global_amp = global_amp.unsqueeze(0).repeat(batch_size, 1)

    if flatten:
        freqs      =      freqs.sum(dim=-1)
        amps       =       amps.sum(dim=-1)
        phases     =     phases.sum(dim=-1)
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
        saturate_global_amp - Whether to use `global_amp` as `logit(a_k)`, enabling [0, 1]-bounded a_k with `sigmoid(global_amp)`
        normalise_complex_grads - Unused feature (always `False`), but keep alive for future experiments
    """

    # Purpose: Keep in [0, 1]
    # This is not general way, but it is correct for the paper's experiment because a reference signal's amplitude α_k is always <1.
    saturate_or_id = torch.sigmoid if saturate_global_amp else lambda x:x

    for batch in dataloader:

        # Preparation

        ## Target: Adjust type and device, extract info
        target_signal = batch["signal"].float().to(device)
        target_freq   = batch["freq"  ].float().to(device)
        target_amp    = batch["amp"   ].float().to(device)
        target_snr    = batch["snr"   ].float()
        target_len    = target_signal.shape[-1]

        ## InitParam: Adjust size (B, K) -> (B', K)
        angle, mag, phase, global_amp = initial_params
        batch_size = target_signal.shape[0]
        angle      =      angle[:batch_size]
        mag        =        mag[:batch_size]
        phase      =      phase[:batch_size]
        global_amp = global_amp[:batch_size]

        ## Param: `angle` & `mag` for real oscillator | `z` & `global_amp` for multi complex oscillator
        optimizer_params = []
        if use_real_sinusoid_baseline:
            angle.requires_grad_(True)
            mag.requires_grad_(True)
            optimizer_params += [angle, mag]
        else:
            # z
            z = mag * torch.exp(1j * angle)
            z.detach_().requires_grad_(True)
            optimizer_params += [z]
            # global_amp
            # TODO: Don't we need `global_amp.requires_grad_(True)` ?
            if use_global_amp:
                global_amp.requires_grad_(True)
                optimizer_params += [global_amp]
            # Initial phase
            initial_phase = torch.exp(1j * phase)

        ## Optimizer
        optimizer = hydra.utils.instantiate(optimizer_cfg, optimizer_params)

        # /Preparation

        # DiffAbS loop
        for step in range(n_steps):
            # Forward
            optimizer.zero_grad()
            if use_real_sinusoid_baseline:
                pred_signal = real_oscillator(angle, mag, phase, target_len).sum(dim=-2)
            else:
                #                                                              afterwards apply global_amp to each partials | single mode has only 1 partial
                pred_signal = complex_oscillator(z, initial_phase, target_len, reduce=False if use_global_amp or mode == "single" else True)
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
            # if normalise_complex_grads and not use_real_sinusoid_baseline:
            #     z.grad = z.grad / torch.clamp(z.grad.abs(), min=1e-10)
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
            # Frequency extraction
            pred_freq = z.angle().abs()
            # Amplitude correction
            pred_amp = hydra.utils.call(amplitude_estimator_cfg, z[..., None], target_len)[..., 0]
            if use_global_amp:
                pred_amp = pred_amp * saturate_or_id(global_amp)
        ## Signal generation with corrected parameters
        pred_signal = real_oscillator(pred_freq, pred_amp, phase, target_len).sum(dim=-2)
        # /Surrogate-to-Sinusoid

        # Evaluation: GroundTruth vs fitted Oscillator transferred from Surrogate
        metrics = hydra.utils.call(
            metric_fn_cfg,
            target_signal.detach(), target_freq.detach(),               target_amp.detach(), target_snr.detach(),
            pred_signal.detach(),   pred_freq.detach() / (2 * math.pi), pred_amp.detach(),
        )
        # /Evaluation

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