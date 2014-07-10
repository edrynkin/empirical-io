import numpy as np
from scipy.optimize import fsolve


def berry_inversion(hats, tildes):
    """
    Berry inversion step: given sample market shares, estimate deltas
    Arguments:
    hats (ndarray, N x 1): market shares in sample
    tildes (function N x 1 -> N x 1): a mapping from deltas to market shares

    Returns:
    delta_hat (ndarray, N x 1): estimated deltas (mean utility vector)
    """
    tildes_minus_hats = lambda delta: tildes(delta) - hats # s(d) = S <=> s(d) - S = 0
    delta_0 = np.zeros_like(hats) # initial approximation
    delta_hat = fsolve(tildes_minus_hats,delta_0) # solving for the root
    return delta_hat


def iv_regression(X,Z,p,delta_hat):
    """
    Given X,Z,p,delta_hat estimate alpha, beta by 2SLS
    Arguments:
    X (ndarray, N x K) -- demand
    Z (ndarray, N x I) -- instruments
    p (ndarray, N x 1) -- prices
    delta_hat (nd_array, N x 1) -- estimated mean utility vector

    Returns:
    alpha (double): coefficient on prices (p)
    beta (ndarray, N x K): coefficients on demand determinants (X)
    """
    eta,_,_,_ = np.linalg.lstsq(Z, p) # first-stage regression
    p_hat = Z.dot(eta) # fitted values
    X_prime = np.c_[X,p_hat] # appending fitted prices from the 1st stage to the data matrix
    theta,_,_,_ = np.linalg.lstsq(X_prime, delta_hat) # second-stage regression
    beta, alpha = np.split(theta, [np.size(theta)-1]) # split the estimated coefficient vector ([1..N-1],[N])
    return alpha, beta


def berry_estimator(X, Z, p, s_hat, s_tilde):
    """
    Given X,Z,p,delta_hat estimate alpha, beta by 2SLS
    Arguments:
    X (ndarray, N x K) -- demand determinants (observed characteristics)
    Z (ndarray, N x I) -- instruments
    p (ndarray, N x 1) -- prices
    s_hat (ndarray, N x 1) -- market shares
    s_tilde (function: N x 1 -> N x 1) -- mapping from deltas to market shares

    Returns:
    alpha (double): coefficient on prices (p)
    beta (ndarray, N x K): coefficients on observed demand-determining characteristics(X)
    """
    delta_hat = berry_inversion(s_hat, s_tilde)
    alpha, beta = iv_regression(X, Z, p, delta_hat)
    return alpha, beta


def logit_shares(delta):
    """
    Given delta, estimate market shares according to multinomial logit
    Arguments:
    delta (ndarray, N x 1): mean utility vector

    Returns:
    shares (ndarray, N x 1): market shares of all firms, excluding the outside good
    """
    exp_delta = np.exp(delta)
    sum_exp_delta = np.sum(exp_delta) + 1
    return exp_delta / sum_exp_delta

def vertical_differentiation():
    pass