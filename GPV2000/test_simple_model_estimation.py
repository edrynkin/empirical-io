# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 18:26:17 2014

@author: evgeni
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from simple_model_estimation import estimate_values_density, estimate_strategy, estimate_full
from kernels import generate_kernel

np.random.seed(52)

V = np.random.rand(1000,2)
B = V/2
kg = generate_kernel('s-kernel',3,2)
rhog = 1
kf = generate_kernel('s-kernel',3,2)
hg = 0.1
hf = 0.1
grid = np.linspace(0,1,1000)
est1 = estimate_values_density(B, kg, hg, rhog, kf, hf, grid)
act1 = np.ones_like(grid)
plt.plot(grid, est1)
plt.plot(grid, act1)
plt.legend(['estimate', 'true'], loc='upper left')
plt.show()
est2 = estimate_strategy(B, kg, hg, grid)
act2 = grid/2
plt.plot(grid, est2)
plt.plot(grid, act2)
plt.legend(['estimate', 'true'], loc='upper left')
plt.show()
est3, est4 = estimate_full(B, kg, hg, rhog, kf, hf, grid)
print np.allclose(est1[np.isfinite(est1)], est3[np.isfinite(est3)], atol=1e-12, rtol=0),\
        np.allclose(est2[np.isfinite(est2)], est4[np.isfinite(est4)], atol=1e-12, rtol=0)