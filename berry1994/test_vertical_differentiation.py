import unittest
import numpy as np
import scipy.stats
from berry1994.basic_demand_estimation import berry_inversion_vd, shares_vd, berry_estimator_vd, vd_d_to_shares, vd_shares_to_d


class TestVerticalDifferentiation(unittest.TestCase):


    def setUp(self):
        self.N = 10
        self.beta = 15
        self.cdf  = lambda x: scipy.stats.uniform.cdf(x,scale=1,loc=0.5)
        self.icdf = lambda x: scipy.stats.uniform.ppf(x,scale=1,loc=0.5)
        self.X, self.Z, self.p, self.s_hat, self.delta = \
            self.generate_data(self.N, self.beta, self.cdf)

    def test_cdf(self):
        self.assertEqual(0.5,self.cdf(self.icdf(0.5)))

    def test_d_to_s(self):
        s = np.array([0.1, 0.2, 0.3, 0.4])
        d = vd_shares_to_d(s,self.cdf,self.icdf)
        spp = vd_d_to_shares(d,self.cdf)
        self.assertTrue(np.allclose(s,spp))

    def test_berry_inversion(self):
        shares = np.array([0.1, 0.2, 0.3, 0.4])
        prices = np.array([2.0, 5.0, 6.0, 8.0])
        delta_hat = berry_inversion_vd(shares, prices, self.cdf, self.icdf)
        s_hat = shares_vd(delta_hat,prices, self.cdf)
        correct = np.allclose(shares,s_hat)
        self.assertTrue(correct)


if __name__ == '__main__':
    unittest.main()