from __future__ import division
from scipy.optimize import minimize
import numpy as np
import itertools

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
    print 'Start AR-algorithm'   
    k = np.shape(lower)[0] # Dimansionality
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
    print 'Start SI'
    return np.mean([f(row) for row in S])