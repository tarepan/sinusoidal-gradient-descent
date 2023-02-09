<!-- I directly edited this README.md -->

<div align="center">

# Sinusoidal Frequency Estimation by Gradient Descent <!-- omit in toc -->
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook] -->
[![Paper](http://img.shields.io/badge/paper-arxiv.2210.14476-B31B1B.svg)][paper]

</div>

Differential sinusoid with the ***exponentially decaying sinusoid*** surrogate (the real part of an exponentiated complex number).  
Wirtinger derivatives of this surrogate in the complex plane lead our optimizer to the correct solution,
so we can use this surrogate for differential oscillator (frequency estimator) module in DDSP.  


## How to Use
First, install from GitHub through pip.
``` bash
!pip install git+https://github.com/tarepan/sinusoidal-gradient-descent -q
```

Now you can generate a signal with the oscillator.  
``` python
from sinusoidal_gradient_descent.core import complex_oscillator

z = torch.exp(1j * torch.rand(1) * math.pi) # Random frequency and amplitude
signal = complex_oscillator(z, N=1024, reduce=True)
```

For frequency estimation (fitting toward a signal), you just loop 'synthesize-loss-backward-optimize' (Diff AbS).  
Runnable demo can be seen in `/estimation.ipynb`.  

The experiments in the paper can be run.
``` bash
python -m sinusoidal_gradient_descent.eval.estimator -cn <setup_name>
```
`<setup_name>` specify experiment setup.  
You can use `single` or `multi_[mse|fft]_[2|8|32](_baseline)` (please check `/estimator_config/`).  


## Implementation variants

Implementation variants exist. There are 3 reference implementations which represent same surrogate.  

| name                     |        ops        | time-varying z | numerical instability | inference time [µs/loop] |
| ------------------------ | ----------------- | -------------- | --------------------- | ------------------------ |
| direct exponentiation    | exp(z)     . real |      -         | amplitude & phase     |      15.0 ± 0.74 µs      |
| cumulative product       | cumprod(z) . real |      ✅       | amplitude & phase     |      18.3 ± 0.90 µs      |
| directly damped sinusoid | exp(a)     * cos  |      -         | amplitude             |      27.2 ± 0.35 µs      |
| -                        | cumprod(a) * cos  |      ✅       | amplitude             |           -              |

Tested on an Intel i5 2GHz Quad Core CPU.
You can check the implementations in `/oscillators.py`.


[paper]: https://arxiv.org/abs/2210.14476
<!-- [notebook]: https://colab.research.google.com/github/tarepan/S3PRL_VC/blob/main/s3prlvc.ipynb -->
