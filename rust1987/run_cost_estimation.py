from data_generation import *
from cost_estimation import *

t_1 = np.array([0.5,0.1,0.7])
RC, theta_3, beta, T, engines = 0.25, 0.75, 0.925, 200, 2
engines = generate_data(t_1, cubic_cost, RC, theta_3, beta, T, engines)
result = estimate_costs(engines, cubic_cost, beta)
print result