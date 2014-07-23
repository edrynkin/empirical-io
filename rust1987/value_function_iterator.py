from __future__ import division
from scipy.optimize import minimize
from scipy.interpolate import griddata
import numpy as np
import itertools

def value_function(x_min = 0, x_max = 5, e_min = -2, e_max = 5, u = (lambda x,i: -0.02*x**2 - i*2), beta = 0.95,
                   num_in_grid = [100, 20, 20], num_sim_data = 1e2, 
                   tolerance = 1e-1, F = None, x0 = None):
    if F == None:
        g = 0.5772 # Euler's constant
        F = lambda x, e1, e2: np.exp(-np.exp(-e1-g))*np.exp(-np.exp(-e2-g))*(1-np.exp(-(x)*(x>0)))
    if x0 == None:
        x0 = np.array([(x_min+x_max)/2, 0, 0]) # initial point for optimization algorithms
    G = lambda x: F(x[0], x[1], x[2])
    f = lambda x: density(G, x)  
    lower = np.array([x_min, e_min, e_min]) # lower bound of rectangular area under consideration
    upper = np.array([x_max, e_max, e_max]) # upper bound --//--
    M = -1.1*(minimize(lambda x: 0-f(x), x0, method='Nelder-Mead')['fun'])*\
                        ((x_max-x_min)*(e_max-e_min)**2) # acceptance-rejection algorithm constant
    V = iterator(u, beta, f, lower, upper, num_in_grid, num_sim_data, tolerance, None, x0, M)
    return V
    
def iteration(u, beta, V, S, x_high):
    EV = lambda y: beta*simulated_integral(lambda x: V(x[0]+np.array(y), x[1]+np.zeros(np.size(y)), x[2]+np.zeros(np.size(y))), S) # discounted EV_{t+1}(x)     
    V0 = lambda x, e0, e1: (u(x,0) + EV(x) + e0)*(x<x_high) # V_t(x, e0, e1|i=0)   
    V1 = lambda x, e0, e1: (u(x,1) + EV(0) + e1)*(x<x_high) # V_t(x, e0, e1|i=1)   
    U = lambda x, e0, e1: np.array([V0(x, e0, e1), V1(x, e0, e1)]).max(axis=0) # V(x, e0, e1)   
    return U
    
def iterator(u, beta, f, lower, upper, num_in_grid, num_sim_data = 1e4, 
             tolerance = 1e-1, x_high = None, x0 = None, M = None):
    if x_high == None:
        x_high = 20*upper[0] # maximal possible value of x
    if x0 == None:
        x0 = (lower + upper)/2
    V = lambda x, e1, e2: 0 # starting point for contraction algorithm
    V = np.vectorize(V)    
    S = acceptance_rejection_algorithm(f, lower, upper, num_sim_data, M, x0) # simulated sample
    m = 2*tolerance # imitial divergence
    iter_num = 0
    xaxis = np.linspace(lower[0], x_high, num_in_grid[0])
    eaxis0 = np.linspace(lower[1], upper[1], num_in_grid[1])
    eaxis1 = np.linspace(lower[2], upper[2], num_in_grid[2])
    X, E0, E1 = np.meshgrid(xaxis, eaxis0, eaxis1, indexing='ij') # grid
    X = X.ravel()
    E0 = E0.ravel()
    E1 = E1.ravel()
    I1 = np.zeros(np.prod(num_in_grid)) 
    while m>tolerance:
        iter_num += 1
        I0 = I1 # values on the grid before iteration
        U = iteration(u, beta, V, S, x_high) # iteration step
        I1 = U(X, E0, E1) # values on the grid after iteration
        V = lambda x, e0, e1: griddata(np.array([X, E0, E1]).T, I1, np.array([x, e0, e1]).T) # interpolation
        m = np.abs(I1-I0).max() # divergence
        print 'iteration ', iter_num, ', divergence ', m
    return V

def density(F, x, tolerance = 1e-1):
    """
    The multivariate density function. Approximates the density of arbitrary
    multivariate distribution givep CDF and point in the n-dimensional space.
    The approximation is given by finite difference procedure.
    
    Inputs:
    F -- the n-dimensional CDF. Function (n,) -> [0,1]
    x -- the n-dimensional array, point at which estimation of density required.
    tolerance -- the finite difference for the approximation.
    
    Outputs:
    f -- the PDF of the distribution at the point x.
    
    Examples:
    In [1]: F = lambda x: np.prod(sp.stats.norm.cdf(x))
    In [2]: density(F, [1,-1,0])
    Out[2]: 0.023260950184178501
    
    """
    f = 0 # Initialization
    k = np.size(x) # Dimensionality
    t = itertools.product(range(2), repeat=k) # Set of all subsamples coded by 1's and 0's
    for z in t: # Inclusion-exclusion estimation procedure
        d = tolerance # Tolerance
        y = np.array(x) + (np.ones(k) - np.array(z))*d # Point for CDF evaluation
        p = sum(z) # Parity
        f = f + (-1)**p*F(y) # Inclusion or exclusion of CDF at the point y.
    f = f/d**k # Normalization by volume of the rectangular
    return f
    
def acceptance_rejection_algorithm(f, lower, upper, num = 1e4, M = None, x0 = None):
    """
    The acceptance-rejection algorithm for simulating the sample from arbitrary
    multivariate distribution given any function proportional to the PDF f(x).
    The algorithm approximates the sample from the actual distribution on the 
    rectangular "lower" <= x <= "upper" using the uniform (on this rectangular) 
    distribution as the basic one. The tails are dropped out.
    
    Inputs:
    f -- function proportional to the CDF. n-by-1 to 1-by-1 function.
    lower -- n-by-1 array of lower bounds of the rectangular sampling area.
    upper -- n-by-1 array of upper bounds of the rectangular sampling area.
    num -- the minimal size of simulated sample.
    M -- the number such that f(x) < M*g(x) for any x in [lower, upper] where 
         g(x) is the uniform distribution on [lower, upper].
    x0 -- the starting point for finding M if M is not given.
    
    Outputs:
    Out -- num-by-n array of num n-dimansional random variables. 
    
    """  
    print 'AR started'
    k = np.size(lower) # Dimansionality
    g = 1/np.prod(upper-lower) # Uniform PDF on [lower, upper]
    if M == None: # Initialization of M
        M = -1.1*(minimize(lambda x: 0-f(x)/g, x0, method='Nelder-Mead')['fun'])
    c = 0 # Initialization of random sample size
    Out = np.empty((0,k)) # Initialization of random sample
    while c<num:
        S = lower + np.random.rand(num*10, k)*(upper-lower) # Sample from g(x)
        U = np.random.rand(np.shape(S)[0]) # Uniform [0,1] random variables 
        F = [f(row) for row in S] # f estimated at random uniform draw
        T = U<F/M/g # Acceptance-rejection step
        Out = np.vstack((Out,S[T])) # Output
        c += T.sum() # Random sample size
        print c
    print 'AR finished'
    return Out
    
def simulated_integral(f, S):
    """
    Simulated integral
    
    Inputs:
    f -- the integrated function. f: k-by-1 -> 1-by-1
    S -- the simulated sample from the distribution of interest. n-by-k array.
    
    Outputs:
    The numerical integral of f w.r.t. distribution of the sample S. A real number.
    
    """
    return np.mean([f(row) for row in S], axis=0)
    
    
#V = value_function()