import numpy as np


def compute_expected_profits(W, X, Y, alpha, beta, gamma, delta, kappa):
    M, _ = W.shape
    signs = -1.0 * np.ones_like(alpha)
    signs[0] = 1.0
    c_alpha = alpha * signs # to get alpha_1 - alpha_2 - alpha_3 - ... - alpha_N after cumsum
    c_alpha = np.cumsum(c_alpha)
    c_gamma = np.cumsum(gamma)
    S = Y.dot(kappa)
    S = np.reshape(S, newshape=(M, 1))
    Xb = X.dot(beta)
    Xb = np.reshape(Xb, newshape=(M, 1))
    V = Xb + c_alpha
    Wd = W.dot(delta)
    Wd = np.reshape(Wd, newshape=(M, 1))
    F = Wd + c_gamma
    SV = S * V
    P = SV - F
    return P


def compute_shapes(alpha, beta, delta, kappa):
    N, KL, J, L = alpha.size, beta.size, kappa.size, delta.size
    return J, KL, L, N


def combine_theta(alpha, beta, gamma, delta, kappa):
    return np.concatenate([alpha, beta, gamma, delta, kappa])