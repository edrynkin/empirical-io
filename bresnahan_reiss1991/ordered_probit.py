import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def generate_data(N, beta, mu):
    K = beta.size
    X = np.random.random((N, K))
    y_star = X.dot(beta) + np.random.standard_normal(N)
    y_star = np.expand_dims(y_star, axis=1)
    neg_inf = float("-inf")
    pos_inf = float("inf")
    mu_ = np.hstack((neg_inf, mu, pos_inf))
    y = np.diff(y_star > mu_)
    y = y.astype(int)
    return X, y


def log_likelihood(X, y, mu, beta):
    neg_inf = float("-inf")
    pos_inf = float("inf")
    mu_ = np.hstack((neg_inf, mu, pos_inf))
    mu_ = np.expand_dims(mu_, axis=1)
    Xb = X.dot(beta)
    phi = norm.cdf(mu_ - Xb)
    dphi = np.diff(phi.T)
    ldphi = np.log(dphi)
    yl = ldphi * y
    log_l = np.sum(yl.ravel())
    return log_l


def estimate(X, y, mu, beta_0):
    """
    > np.random.seed(3)
    > beta = np.ones(15)
    > mu = np.array([5.0, 6.5, 8.7])
    > X,y = generate_data(1000, beta, mu)
    > beta_0 = 0.4*beta
    > beta_hat = estimate(X, y, mu, beta_0)
    > print beta_hat
    """
    f = lambda beta: -1.0 * log_likelihood(X, y, mu, beta)
    result = minimize(f, beta_0, method="Nelder-Mead", jac=False,
                      tol=1e-10,
                      options={"maxiter": 2000})
    return result["x"]

