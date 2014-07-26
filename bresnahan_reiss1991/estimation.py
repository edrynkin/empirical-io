import numpy as np
from openopt.kernel.baseProblem import maximize
from scipy.optimize import fsolve, minimize
from common import compute_expected_profits
from scipy.stats import norm


def extract_shapes(W, X, Y):
    M, KL = X.shape
    _, J = Y.shape
    _, L = W.shape
    return J, KL, L, M


def log_likelihood(W, X, Y, n, N, theta):
    J, KL, L, M = extract_shapes(W, X, Y)
    alpha, beta, gamma, delta, kappa, _ = split_theta(theta, N, KL, L, J)
    P = compute_expected_profits(W, X, Y, alpha, beta, gamma, delta, kappa)
    pos_inf = float("inf")
    neg_inf = float("-inf")
    P = np.c_[pos_inf * np.ones(M), P, neg_inf * np.ones(M)]
    probabilities = norm.cdf(P)
    probabilities = np.fliplr(np.diff(np.fliplr(probabilities), axis=1))
    indices = n.astype(int) - 1
    p = np.choose(indices, probabilities.T)
    log_l = np.sum(np.log(p))
    return -1.0 * log_l


def split_theta(theta, N, KL, L, J):
    indices = np.array([N, KL, N, L, J])
    indices = np.cumsum(indices)
    result = tuple(np.split(theta, indices))
    return result


def estimate_by_mle(W, X, Y, n, N, theta_0):
    f = lambda theta: log_likelihood(W, X, Y, n, N, theta)
    J, KL, L, M = extract_shapes(W, X, Y)
    bnds = tuple([(0.0, None) for t in theta_0])
    result = minimize(f, theta_0, method="tnc", jac=False,
                      options={"maxiter": 10000},
                      bounds=bnds)
    theta_hat = result["x"]
    return list(split_theta(theta_hat, N, KL, L, J))