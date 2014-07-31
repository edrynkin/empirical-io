# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 15:25:42 2014

@author: evgeni
"""
from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal, chi2
import matplotlib.pyplot as plt
from kernels import density_estimation, conditional_mean_estimation, generate_kernel, check_kernel_order

np.random.seed(52)

k = generate_kernel('s-kernel', 1, 6)
print check_kernel_order(k,2), ' ', check_kernel_order(k,4), ' ', check_kernel_order(k,6), ' ', check_kernel_order(k,8) 
k = generate_kernel('s-kernel', 1, 2)
X = np.random.chisquare(5, 1000)
h = (np.max(X, 0) - np.min(X , 0))/20
grid1 = np.linspace(X.min(), X.max(), 1000)
est1 = density_estimation(X, k, h, grid1)
act1 = np.reshape(chi2.pdf(grid1, 5), (1000,1))
mu = [0, 0]
sigma = [[1, 0.5], [0.5, 2]]
X = np.random.multivariate_normal(mu, sigma, 1000)
h = (np.max(X, 0) - np.min(X , 0))/20
x_grid, y_grid = np.meshgrid(np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100),
                             np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100))
grid2 = np.vstack((x_grid.ravel(),y_grid.ravel())).T
est2 = density_estimation(X, k, h, grid2)
s = multivariate_normal(mu, sigma)
act2 = np.reshape(s.pdf(grid2), (10000,1))

plt.plot(grid1, est1)
plt.plot(grid1, act1)
plt.legend(['estimate', 'true'], loc='upper right')
plt.show()

X = np.random.rand(10000)*10
E = np.random.randn(10000)*2
Y = np.sqrt(X) + E
grid3 = np.linspace(0,10,1000)
h = 0.5
est3 = conditional_mean_estimation(X, Y, k, h, grid3)
act3 = np.reshape(np.sqrt(grid3), (1000,1))
plt.plot(grid3, est3)
plt.plot(grid3, act3)
plt.legend(['estimate', 'true'], loc='upper left')
plt.show()