from data_generation import *
from cost_estimation import *

t_1 = np.array([0.5,0.25,0.75])
RC, theta_3, beta, T, engines = 0.25, 0.75, 0.925, 100, 2
engines = generate_data(t_1, cubic_cost, RC, theta_3, beta, T, engines)
print estimate_costs(engines, cubic_cost, beta)
