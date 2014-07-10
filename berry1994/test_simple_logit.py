import numpy as np
import unittest
from berry1994.basic_demand_estimation import berry_inversion, logit_shares, iv_regression, berry_estimator


class TestSimpleLogit(unittest.TestCase):

    @staticmethod
    def generate_data(n, i, k, beta, alpha, s_tilde):
        np.random.seed(32)
        eps = 1e-1
        gamma = np.ones(i)
        Z = np.random.randn(n,i)
        p = Z.dot(gamma)
        p = p + eps*np.random.randn(p.size)
        X = np.random.randn(n,k)
        delta = X.dot(beta) + alpha * p
        delta = delta + eps*np.random.randn(delta.size)
        s_hat = s_tilde(delta)
        return X, Z, p, s_hat, s_tilde, delta

    def setUp(self):
        self.N = 10
        self.I = 2
        self.K = 3
        self.beta = [1, 2, 4]
        self.alpha = -2
        self.X, self.Z, self.p, self.s_hat, self.s_tilde, self.delta = \
            self.generate_data(self.N, self.I, self.K, self.beta, self.alpha, logit_shares)


    def test_berry_inversion(self):
        shares = np.array([0.4, 0.3, 0.2, 0.1])
        delta_hat = berry_inversion (shares, logit_shares)
        correct = np.allclose(shares,logit_shares(delta_hat) )
        self.assertTrue(correct)

    def test_berry_inversion2(self):
        delta_hat = berry_inversion (self.s_hat, logit_shares)
        correct = np.allclose(self.s_hat,logit_shares(delta_hat),atol=0.1,rtol=0.1)
        self.assertTrue(correct)

    def test_instrumental_variables(self):
        alpha_hat, beta_hat = iv_regression(self.X,self.Z,self.p,self.delta)
        alpha_close = np.allclose(self.alpha, alpha_hat, atol=0.1, rtol=0.1)
        beta_close = np.allclose(self.beta, beta_hat, atol=0.1, rtol=0.1)
        self.assertTrue(alpha_close)
        self.assertTrue(beta_close)

    def test_berry_estimator(self):
        alpha_hat, beta_hat = berry_estimator(self.X,self.Z,self.p,self.s_hat,self.s_tilde)
        alpha_close = np.allclose(self.alpha, alpha_hat, atol=0.1, rtol=0.1)
        beta_close = np.allclose(self.beta, beta_hat, atol=0.1, rtol=0.1)
        self.assertTrue(alpha_close)
        self.assertTrue(beta_close)


