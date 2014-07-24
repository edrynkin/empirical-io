import itertools
import numpy as np
from common import expected_value, utility


def estimate_transition_probability(engines):
    s = 0.0
    n = 0.0
    for engine in engines:
        x = engine[0]
        i = engine[1]
        for t in xrange(i.size-1):
            increase = x[t+1] - (1 - i[t]) * x[t]
            s += increase
            n += 1.0
    return s/n


def logit(u, beta, EV):
    result = np.exp(u + beta*EV)
    s = np.sum(result, axis=1)
    s = np.expand_dims(s,axis=1)
    result /= s
    result = np.log(result)
    return result


def estimate_costs(engines, cost_func, beta):
    theta_3 = estimate_transition_probability(engines)
    grid_t = np.mgrid[0:1.1:0.25, 0:1.1:0.25, 0:1.1:0.25]
    dimensions = grid_t.shape
    d = dimensions[0]
    N = dimensions[1]
    grid_RC = np.mgrid[0:1.1:0.25]
    max_log_likelihood = float("-inf")
    arg_max_log_likelihood = None
    iter_number = 0
    for theta_1_index in itertools.product(xrange(N), repeat=d):
        iter_number += 1
        if iter_number % 10 == 0:
            print "Iteration", iter_number
        i, j, k = theta_1_index
        theta_1 = grid_t[:, i, j, k]
        for RC in np.nditer(grid_RC):
            log_likelihood = 0.0
            EV = expected_value(theta_3, RC, cost_func, theta_1, beta)
            for engine in engines:
                x = engine[0]
                i = engine[1]
                u = utility(x, RC, cost_func, theta_1)
                log_probability_measure = logit(u, beta, EV[x.astype(int)])
                log_probability = log_probability_measure[:, i.astype(int)]
                log_likelihood += np.sum(log_probability)
            if max_log_likelihood < log_likelihood:
                max_log_likelihood = log_likelihood
                print max_log_likelihood
                arg_max_log_likelihood = (theta_1, RC)
                print arg_max_log_likelihood
    t1, rc = arg_max_log_likelihood
    return t1, rc, theta_3