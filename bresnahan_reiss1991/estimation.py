import numpy as np
from scipy.optimize import minimize
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
    P = np.c_[pos_inf * np.ones(M), P, neg_inf * np.ones(M)] # pad with positive and negative infinities
    probabilities = norm.cdf(P) # compute F (F (neg_inf) = 0, F (pos_inf) = 1)
    probabilities = np.fliplr(np.diff(np.fliplr(probabilities), axis=1)) # fliplr to get F(P_N) - F(P_{N+1})
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
    '''
    Estimate ordered probit model by conditional MLE.
    Truncated Newton method is used to optimize the log-likelihood function.

    Inputs:
    W: M x L matrix of cost shifters
    X: M x (K + L) matrix of demand and cost shifters
    Y: M x J matrix of market size determinants
    n: M x 1 vector of firm numbers
    N: double -- maximum possible number of firms
    theta_0: (2*N+K+L+L+J)x1 -- initial approximation to the parameter value

    Outputs:
    alpha: N x 1 vector of all variable costs competition parameters
    beta: (K + L) x 1 vector of variable costs parameters
    gamma: N x 1 vector of all fixed costs competition parameters
    delta: L x 1 vector of all
    kappa: J x 1 vector of all market size parameters
    '''
    f = lambda theta: log_likelihood(W, X, Y, n, N, theta)
    J, KL, L, M = extract_shapes(W, X, Y)
    bnds = tuple([(0.0, None) for t in theta_0])
    result = minimize(f, theta_0, method="tnc", jac=False,
                      tol=1e-8,
                      options={"maxiter": 30000},
                      bounds=bnds)
    theta_hat = result["x"]
    return list(split_theta(theta_hat, N, KL, L, J))