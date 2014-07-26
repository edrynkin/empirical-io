import numpy as np
from common import compute_expected_profits, compute_shapes


def generate_data(alpha, beta, gamma, delta, kappa, M):
    '''
    Generate data for the Bresnahan-Reiss model.
    Structural errors are drawn from N(0,1)

    Inputs:
    alpha: N x 1 vector of all variable costs competition parameters
    beta: (K + L) x 1 vector of variable costs parameters
    gamma: N x 1 vector of all fixed costs competition parameters
    delta: L x 1 vector of all
    kappa: J x 1 vector of all market size parameters
    M: double, number of markets

    Outputs:
    n: M x 1 vector of the firm numbers
    W: M x L matrix of cost determinants
    X: M x (K + L) matrix of cost and demand shifters
    Y: M x (J + 1) matrix of market size determinants
    Z: M x K matrix of demand shifters
    '''
    J, K, KL, L, N = compute_shapes(alpha, beta, delta, kappa)
    Y = np.random.uniform(size=(M, J + 1), low=0.02, high=0.1)
    X = np.random.uniform(size=(M, KL), low=0.02, high=0.1)
    W = X[:, K:]
    Z = X[:, :K]
    P = compute_expected_profits(W, X, Y, alpha, beta, gamma, delta, kappa)
    eps = np.random.normal(size=(M, 1))
    P = P + eps# find the last positive element in an array
    n = np.sum(P > 0, axis=1) # apply to all rows
    return n, W, X, Y, Z
