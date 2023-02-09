import torch

from .oscillators import complex_oscillator_direct, complex_oscillator_cumprod, complex_oscillator_damped


# Configs
N, freq = 64, torch.tensor(0.7)

# Inputs & Ground-Truth
z = torch.exp(1j * freq)
cosine_reference = torch.cos(freq * torch.arange(N))


def test_complex_oscillator_direct():
    """Test sinusoid generation by `complex_oscillator_direct`."""

    torch.testing.assert_close(complex_oscillator_direct(z, N),  cosine_reference)


def test_complex_oscillator_cumprod():
    """Test sinusoid generation by  `complex_oscillator_cumprod`."""

    torch.testing.assert_close(complex_oscillator_cumprod(z, N), cosine_reference)


def test_complex_oscillator_damped():
    """Test sinusoid generation by `complex_oscillator_damped`."""

    torch.testing.assert_close(complex_oscillator_damped(z, N),  cosine_reference)
