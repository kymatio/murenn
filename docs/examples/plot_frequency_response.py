# coding: utf-8
"""
=========================
Frequency Magnitude Response
=========================

This notebook demonstrates how to visualize the frequency response
of a 1-D convolutional layer (Conv1D).
"""

#########################
# Standard imports
import torch
from matplotlib import pyplot as plt
import numpy as np

import murenn
#########################################
# number of scales
J = 8

# number of conv1d filters per scale
Q = 10

# kernel size
T = 10

# input channel
C = 1

# output signal length
N = 2 ** J * T

# MuReNN
tfm = murenn.MuReNNDirect(J=J, Q=Q, T=T, in_channels=C)
w = tfm.to_conv1d.detach()
w = w.view(1, J, Q, -1)

#########################################################################
# Plot the spectrums per scale per filter.
colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
plt.figure(figsize=(10, 3))
for q in range(Q):
    for j in range(J):
        w_hat = torch.fft.fft(w[0, j, q, :])
        plt.semilogx(torch.abs(w_hat), color=colors[j])
plt.grid(linestyle='--', alpha=0.5)
plt.xlim(0, N//2)
plt.xlabel("Frequency")
plt.title(f'murenn v{murenn.__version__}. Frequency Magnitude Response')
plt.show()