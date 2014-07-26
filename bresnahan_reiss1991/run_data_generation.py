import itertools
from common import combine_theta
from estimation import estimate_by_mle
from data_generation import *
import numpy as np

np.random.seed(6)
alpha = np.array([5.0, 0.25, 0.75])
beta = np.array([1.0, 2.0, 5.0, 7.0])
gamma = np.array([1.0, 2.0, 3.0])
delta = np.array([1.0, 2.0])
kappa = np.array([1.0, 7.0, 8.0])
actual_params = [alpha, beta, gamma, delta, kappa]
M = 400
N = alpha.size
n, W, X, Y, Z = generate_data(alpha, beta, gamma, delta, kappa, M)
theta_0 = combine_theta(alpha, beta, gamma, delta, kappa)
theta_0 += 0.001 * np.random.random(theta_0.shape)
estimated_params = estimate_by_mle(W, X, Y, n, N, theta_0)
for a, e in itertools.izip(actual_params, estimated_params):
    print a, " ", e