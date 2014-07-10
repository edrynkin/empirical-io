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
    tildes_minus_hats = lambda delta: tildes(delta) - hats  # s(d) = S <=> s(d) - S = 0
    delta_0 = np.zeros_like(hats)  # initial approximation
    delta_hat = fsolve(tildes_minus_hats, delta_0)  # solving for the root
    return delta_hat


def iv_regression(X, Z, p, delta_hat):
    """
    Given X,Z,p,delta_hat estimate alpha, beta by 2SLS
    Arguments:
    X (ndarray, N x K) -- demand
    Z (ndarray, N x I) -- instruments
    p (ndarray, N x 1) -- prices
    delta_hat (ndarray, N x 1) -- estimated mean utility vector

    Returns:
    alpha (double): coefficient on prices (p)
    beta (ndarray, N x K): coefficients on demand determinants (X)
    """
    eta, _, _, _ = np.linalg.lstsq(Z, p)  # first-stage regression
    p_hat = Z.dot(eta)  # fitted values
    X_prime = np.c_[X, p_hat]  # appending fitted prices from the 1st stage to the data matrix
    theta, _, _, _ = np.linalg.lstsq(X_prime, delta_hat)  # second-stage regression
    beta, alpha = np.split(theta, [np.size(theta)-1])  # split the estimated coefficient vector ([1..N-1],[N])
    return alpha, beta


def berry_estimator(X, Z, p, s_hat, s_tilde):
    """
    Given X,Z,p,delta_hat estimate alpha, beta using Berry's procedure
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
    Given delta, estimate market shares according to multinomial logit model
    Arguments:
    delta (ndarray, N x 1): mean utility vector

    Returns:
    shares (ndarray, N x 1): market shares of all firms, excluding the outside good
    """
    exp_delta = np.exp(delta)
    sum_exp_delta = np.sum(exp_delta) + 1
    return exp_delta / sum_exp_delta

def vd_d_to_shares(d, cdf):
    """
    Auxiliary function: convert cut-off points to shares
    Arguments:
    d(ndarray, N x 1): cut-off points

    Returns:
    shares(ndarray, N x 1): market shares
    """
    dF = cdf(d)
    dF = np.append(dF, [1.0])
    result = np.diff(dF)
    return result


def vd_shares_to_d(s, cdf, i_cdf):
    """
    Auxiliary function: convert shares to cut-off points

    Arguments:
    s (ndarray, N x 1): market shares

    Returns:
    d (ndarray, N x 1): cut-off points
    """
    d = np.zeros(s.size)
    d[-1] = i_cdf(1 - s[-1])
    for j in xrange(d.size-2, -1, -1):
        d[j] = i_cdf(cdf(d[j+1])-s[j])
    return d

def berry_inversion_vd(shares, prices, cdf, i_cdf):
    '''
    Berry inversion step for the Shaked-Sutton vertical differentiation model (section 4, example 2)
    Arguments:
    shares (ndarray, N x 1): market shares
    prices (ndarray, N x 1): product prices
    cdf (function, double -> double): cdf of the taste parameter (v) for the consumers
    icdf (function, double -> double): inverse cdf of the taste parameter (v) for the consumers

    Returns:
    delta(ndarray, N x 1): mean utility levels vector
    '''
    d = vd_shares_to_d(shares, cdf, i_cdf)
    dp = np.diff(np.insert(prices, 0, 0))
    psi = np.divide(dp, d)
    psi = np.cumsum(psi)
    delta = psi - prices
    return delta


def shares_vd(delta, prices, cdf):
    '''
    Given delta, prices, and taste parameter cdf, compute shares.
    Vertical differentiation model is used to specify demand (section 4, example 2).
    Arguments:
    delta (ndarray, N x 1): mean utility levels vector
    prices (ndarray, N x 1): product prices vector
    cdf (function, double -> double): cdf of the taste parameter (v) for the consumers

    Returns:
    shares (ndarray, N x 1): vector of computed market shares
    '''
    dp = np.diff(np.insert(prices, 0, 0))
    psi = delta+prices
    dps = np.diff(np.insert(psi, 0, 0))
    d = np.divide(dp, dps)
    s_hat = vd_d_to_shares(d, cdf)

    return s_hat


def berry_estimator_vd(X, Z, p, shares, cdf, i_cdf):
    """
    Given X,Z,p,cdf,i_cdf estimate alpha, beta by using Berry's procedure.

    Arguments:
    X (ndarray, N x K) -- demand determinants (observed characteristics)
    Z (ndarray, N x I) -- instruments
    p (ndarray, N x 1) -- prices
    cdf (function, double -> double): cdf of the taste parameter (v) for the consumers
    icdf (function, double -> double): inverse cdf of the taste parameter (v) for the consumers

    Returns:
    alpha (double): coefficient on prices (p)
    beta (ndarray, N x K): coefficients on observed demand-determining characteristics(X)
    """
    delta_hat = berry_inversion_vd(shares, p, cdf, i_cdf)
    alpha, beta = iv_regression(X, Z, p, delta_hat)
    return alpha, beta
