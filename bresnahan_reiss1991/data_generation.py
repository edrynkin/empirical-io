import numpy as np
from common import compute_expected_profits, compute_shapes


def generate_data(alpha, beta, gamma, delta, kappa, M):
    J, KL, L, N = compute_shapes(alpha, beta, delta, kappa)
    K = KL - L
    Y = np.random.uniform(size=(M, J), low=0.02, high=0.1)
    X = np.random.uniform(size=(M, KL), low=0.02, high=0.1)
    W = X[:, K:]
    Z = X[:, :K]
    P = compute_expected_profits(W, X, Y, alpha, beta, gamma, delta, kappa)
    eps = np.random.normal(size=(M, 1))
    P = P + eps
    ss = lambda x: N - np.searchsorted(a=x[::-1], v=+0.0)
    n = np.apply_along_axis(arr=P, func1d=ss, axis=1)
    return n, W, X, Y, Z
