import numpy as np

MAX_MILEAGE = 10


def transition(x, i, theta_3):
    mileage = np.random.binomial(1, theta_3)
    return min((1 - i)*x + mileage, MAX_MILEAGE)


def utility(x, RC, cost_func, theta_1):
    zx = np.zeros_like(x)
    u = np.vstack((-1*cost_func(x, theta_1), (-1*(RC+cost_func(zx, theta_1)))))
    return u.T


def control(x_t, eps_t, theta_1, RC, cost_func, beta, EV):
    u_t = utility(np.array([x_t]), RC, cost_func, theta_1).T
    ev = EV[x_t].ravel()
    u_t = u_t.ravel()
    v_t = u_t + eps_t + beta * ev
    i_t = np.argmax(v_t)
    return i_t


def transition_probability_distribution(x_t, i_t, theta_3):
    result = np.zeros(MAX_MILEAGE+1)
    x_t1 = x_t * (1-i_t)
    result[x_t1] += (1-theta_3)
    result[min(x_t1+1, MAX_MILEAGE)] += theta_3
    return result


def expected_value(theta_3, RC, cost_func, theta_1, beta):
    EV_0 = np.zeros((MAX_MILEAGE+1, 2))
    EV_1 = np.zeros((MAX_MILEAGE+1, 2))
    X = np.array(list(range(MAX_MILEAGE+1)))
    run_iteration = True
    num_iter = 0
    while run_iteration:
        for i in xrange(2):
            for x in X:
                p = transition_probability_distribution(x, i, theta_3)
                u = utility(X, RC, cost_func, theta_1)
                u = u + beta*EV_0
                u = np.exp(u)
                u = np.log(np.sum(u, 1))
                value = u.dot(p)
                EV_1[x][i] = value
        run_iteration = not np.allclose(EV_1, EV_0)
        EV_0 = np.copy(EV_1)
    return EV_1


def cubic_cost(x, theta_1):
    x_p = np.vstack((x,x**2,x**3))
    result = theta_1.dot(x_p)
    return result
