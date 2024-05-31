# coding: utf-8
"""
=========================
Effective receptive field
=========================

Following [1], we compute the effective receptive field of a 1-D convolutional layer (Conv1D)
as the magnitude of the gradient of the output at time zero with respect to the output.

Then, we do the same for a MuReNN layer comprising a direct dual-tree complex wavelet
transform (DTCWT) followed by 1-D convolution and inverse DTCWT (IDTCWT):
    DTCWT -> Conv1D -> IDTCWT

We will see that the effective receptive field of MuReNN is larger than that of Conv1D, given
the same parameter budget. This is due to the recursive subsampling in the DTCWT, which has
the effect of dilating the receptive field of the Conv1D operator.

.. [1] Luo, W., Li, Y., Urtasun, R., & Zemel, R. (2016).
    Understanding the effective receptive field in deep convolutional neural networks.
    Advances in neural information processing systems, 29.
"""

################################################################################
# Effective receptive field
import math
import matplotlib.pyplot as plt
import numpy as np
import torch

import murenn


################################################################################
# MuReNN
def effective_receptive_field_murenn(J, T, N, weight_fn):
    """
    Compute the effective receptive field of a MuReNN layer.
        DTCWT -> Conv1D -> IDTCWT
    
    Parameters
    ----------
    J : int
        number of scales
    T : int
        kernel size
    N : int
        input size
    weight_fn : function
        weight initialization function with prototype
        w = weight_fn(in_channels, out_channels, kernel_size)
    
    Returns
    -------
    g : np.ndarray
        effective receptive field
    """
    x = torch.zeros(1, 1, N, requires_grad=True)
    dtcwt = murenn.DTCWT(J=J)
    idtcwt = murenn.DTCWTInverse(J=J)
    x_phi, x_psis = dtcwt(x)
    y_psis = []
    for j in range(J):
        x_j = x_psis[j]
        w = weight_fn(1, 1, T) * math.sqrt(2**j)
        y_j_re = torch.nn.functional.conv1d(x_j.real, w, padding='same')
        y_j_im = torch.nn.functional.conv1d(x_j.imag, w, padding='same')
        y_j = torch.complex(y_j_re, y_j_im)
        y_psis.append(y_j)

    y = idtcwt(yh=y_psis, yl=x_phi)
    y0 = y[0, :, N//2]
    y0.backward()
    g = torch.abs(x.grad.squeeze())**2
    return g


################################################################################
# Simulation

# signal length
N = 2**16
t = torch.arange(1, 1+N//2)

# number of scales
J = 12

# kernel size
T = 7

# number of random i.i.d. samples
B = 2**5

# machine precision
epsilon = torch.finfo(torch.float32).eps

# quantiles
quantiles = torch.Tensor([0.25, 0.75])

murenn_loggrads = []
for b in range(B):
    grad = effective_receptive_field_murenn(J, T, N, torch.randn)
    loggrad = torch.log2(torch.abs(grad))
    murenn_loggrads.append(loggrad)
murenn_loggrads = torch.stack(murenn_loggrads)
murenn_median_loggrad = torch.median(murenn_loggrads, axis=0).values
murenn_quantile_loggrad = torch.quantile(murenn_loggrads, quantiles, dim=0)
murenn_param_count = J * T

################################################################################
# Test
deviation = murenn_median_loggrad[N//2:] + torch.log2(t)
#assert torch.abs(deviation).max() < J, deviation.max()


################################################################################
# Log-log plot
plt.figure(figsize=(5, 5))
plt.plot(torch.log2(t), murenn_median_loggrad[N//2:],
    label=f'MuReNN ({murenn_param_count} parameters)')
plt.fill_between(torch.log2(t), murenn_quantile_loggrad[0, N//2:],
    murenn_quantile_loggrad[1, N//2:], alpha=0.3)

plt.plot(torch.log2(t), -torch.log2(t), linestyle='--', label='Power law')
plt.plot([J+math.log2(T)] * 2, [math.log2(epsilon), 0],
    linestyle='--', label='Theoretical bound')

plt.grid(linestyle='--')
plt.ylabel('Gradient magnitude')
plt.xlabel('Lag')
plt.legend()
plt.gca().set_xticks(range(0, int(math.log2(N)), 2))
plt.gca().set_yticks(range(int(math.log2(epsilon)/2)*2, 2, 2))
plt.gca().set_xticklabels([f'$2^{{{int(x)}}}$' for x in plt.gca().get_xticks()])
plt.gca().set_yticklabels([f'$2^{{{int(x)}}}$' for x in plt.gca().get_yticks()])
plt.gca().set_ylim(int(math.log2(epsilon)), 0)
plt.title(f'murenn v{murenn.__version__}. Effective receptive field')
plt.show()