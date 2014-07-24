import numpy as np
from common import *

def generate_data(theta_1, cost_func, RC, theta_3, beta, T, engines):
    EV = expected_value(theta_3, RC, cost_func, theta_1, beta)
    result = []
    for m in xrange(engines):
        x = np.zeros(T)
        i = np.zeros(T)
        epsilon = np.random.gumbel(0.0, 1.0, (T, 2))
        x_t, i_t = 0.0, 0.0
        for t in xrange(T):
            eps_t = epsilon[t]
            x[t] = transition(x_t, i_t, theta_3)
            i[t] = control(x[t], eps_t, theta_1, RC, cost_func, beta, EV)
            x_t, i_t = x[t], i[t]
        result.append((x,i))
    return result