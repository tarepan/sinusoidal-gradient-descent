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
import math, torch, matplotlib.pyplot
from sinusoidal_gradient_descent.core import complex_oscillator

z = torch.tensor([0.7 + 0.7j]) # complex number, representing 'amplitude' and 'frequency'
signal = complex_oscillator(z, length=100).sum(dim=-2)
matplotlib.pyplot.plot(signal)
```

![surrogate](/doc/img/single_surrogate.png)

For frequency estimation (fitting toward a signal), you just loop 'synthesize-loss-backward-optimize' (Diff AbS).  
Runnable demo can be seen in [/estimation.ipynb][estimation_nb_file] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][estimation_nb_colab]  

The experiments in the paper can be run.
``` bash
python -m sinusoidal_gradient_descent.eval.estimator -cn <conf>
```
`<conf>` specify an experiment configuration.  
You can use `single` or `multi_[mse|fft]_[2|8|32](_baseline)` (please check [/estimator_config/](https://github.com/tarepan/sinusoidal-gradient-descent/tree/main/estimator_config)`).  


## Implementation variants

Implementation variants exist. There are 3 reference implementations which represent same surrogate.  

| name                     |        ops        | time-varying z | numerical instability | inference time [µs/loop] |
| ------------------------ | ----------------- | -------------- | --------------------- | ------------------------ |
| direct exponentiation    | exp(z)     . real |      -         | amplitude & phase     |      15.0 ± 0.74 µs      |
| cumulative product       | cumprod(z) . real |      ✅       | amplitude & phase     |      18.3 ± 0.90 µs      |
| directly damped sinusoid | exp(a)     * cos  |      -         | amplitude             |      27.2 ± 0.35 µs      |
| -                        | cumprod(a) * cos  |      ✅       | amplitude             |           -              |

Tested on an Intel i5 2GHz Quad Core CPU.  
You can check the implementations in [/sinusoidal_gradient_descent/oscillators.py](https://github.com/tarepan/sinusoidal-gradient-descent/blob/main/sinusoidal_gradient_descent/oscillators.py).


[paper]: https://arxiv.org/abs/2210.14476
[estimation_nb_file]: https://github.com/tarepan/sinusoidal-gradient-descent/blob/main/estimation.ipynb
[estimation_nb_colab]: https://colab.research.google.com/github/tarepan/sinusoidal-gradient-descent/blob/main/estimation.ipynb
<!-- [notebook]: https://colab.research.google.com/github/tarepan/S3PRL_VC/blob/main/s3prlvc.ipynb -->
