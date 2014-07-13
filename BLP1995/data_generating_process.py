# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 20:39:46 2014

@author: evgeni

GENERAL DESCRIPTION

Function generate_data generates the test data for BLP (1995) model. The data 
generated is the data for the demand and mark-ups estimations. No data for costs 
estimation is provided. All the simulations use multivariate normal random vectors. 
The generator works as follows. Taking the parameters of distributions as inputs
it generates J-by-K random observables, J random unobservables and J random 
prices first. The nest step is to simulate individual's discrete choice. From
this step market shares (a J-by-1 vector) are obtained. Finally, the shares for 
close values of price vectors are estimated to get the finite difference 
approximation of the ds_(j)/dp_(j) (a J-by-1 vector). Given these approximations
of s(p) and ds/dp the approximations for mark-ups (a J-by-1 vector) are estimated.

IMPORTANT NOTE 

Since no game is solved during the simulations, the algorythm doesn't guarantee 
the existence of the log-linear MC function 
log(mc_j) = w_j*gamma + omega_j (3.1, p.853 in BLP) that will generate simulated
prices. However, it is not necessry for the estimation of the demand for the 
marginal costs to have this specification to have this form. The existence of SOME 
marginal cost function that will lead to such behavior in the equilibrium can
be easily guaranteed by the use of Dirac's delta-function. The equilibrium MC
values will be exactly ones approximated on the last step of the data generation.

SIMULATED SPECIFICATIONS

U_(ij) = x_j * beta - alpha*p_j + xi_j + sum_(k) sigma_k * x_(jk) * v_(jk) + e_(ij),
i.e., (2.5, p.848) from BLP. Where:
x_j is normally distributed vector of observables of j-th good (K-by-1)
beta is mean of random coefficients (K-by-1)
p_j is normally distributed price of j-th good (1-by-1)
alpha is the slope of the utility (1-by-1)
xi_j is normally distributed unobservable component of j-th good (1-by-1)
v_i is standard normally distributed coefficient shocks (K-by-1)
e_(ij) is normally distributed individual-product specific shock
Each individual chooses the j-th alternative iff the utility derived from j-th 
alternative is larger than from all other alternatives and exceeds the reservation
utility level U_0. The probability that individual chooses j-th product given 
x, xi, p and parameters theta is denoted as s_j(x, xi, p, theta). The total output
of good j is then given as q_j(x, xi, p, theta) = s_j(x, xi, p, theta)*N.

The specification of the relation between p and MC is given by (3.3, p.853 in BLP)
for J_f = {j}, i.e., single good producers:
s_j(x, xi, p, theta) + (p_j-MC_j) * ds_j(x, xi, p, theta)/dp_j = 0
 
"""

from __future__ import division
import numpy as np


def generate_data(N = int(1e6), J = 50, K = 3, Mean_Xi = 10, Mean_p = 10, 
                  alpha = 2, Std_e = 5, Reservation_utility = 0, 
                  parsing_size = 10000, Mean_X = None, beta = None, 
                  Sigma_X = None, Sigma_Xi_p = None, Sigma_v = None):
    """
    The data generator

    Inputs: 
    N -- size of the market (a positive integer number)
    J -- number of products (a positive integer number)
    K -- number of observable characteristics (a positive integer number)
    Mean_Xi -- mean of unobservable (a real number)
    Mean_p -- mean of the price (a positive real number)
    alpha -- slope of the utility (a positive real number)
    Std_e -- standard deviation of the random utility (a positive real number)
    Reservation_utility -- reservation utility (a real number)
    parsing size -- size of the blocks used to parse the data (a positive integer number)
    Mean_X -- mean of the observables (a real (K,) vector)
    beta -- mean of the random coefficients (a real (K,) vector)
    Sigma_X -- covariance matrix of the observables (a positively definite K-by-K matrix)
    Sigma_Xi_p -- covariance matrix of the unobservable-price pair (a positively definite 2-by-2 matrix)
    Sigma_v -- covariance matrix of random coefficients (a positively definite K-by-K matrix)
    
    Outputs:
    Output is the J-by-(K+3) matrix such that:
    1) the first column is the prices of each product;
    2) the second column is the shares of each product;
    3) 3-rd to J+2-nd columns are product observable characteristics;
    4) J_3-rd column is the column of at the equilibrium marginal costs (unobservable).
    
    Examples:
    A = generate_data()
    B = generate_data(N=int(1e7))
    C = generate_data(J=200, K=5, Mean_p=30, Mean_X=np.zeros((5,)), beta=np.ones((5,))
                      Sigma_X=np.eye(5), Sigma_v=np.eye(5))
    
    """
    # Default initialization of the variables
    if Mean_X == None:
        Mean_X = np.zeros((K,))
    if beta == None:
        beta = np.ones((K,))
    if Sigma_X == None:
        Correlations_X = np.eye(K) # Correlation structure of observable characteristics
        Std_X = np.ones((K,)) # Standard deviations of observable characteristics
        Sigma_X = (np.eye(K) * Std_X).dot(Correlations_X).dot(np.eye(K) * Std_X)
    if Sigma_Xi_p == None:
        Std_Xi = 2 # Standard deviation of the unobservable characteristics
        Std_p = 1 # Standard deviation of the prices
        Correlation_Xi_p = 0.9 # Correlation between unobservable characteristics and price
        Sigma_Xi_p = np.array([[Std_Xi**2, Correlation_Xi_p*Std_Xi*Std_p],
                        [Correlation_Xi_p*Std_Xi*Std_p, Std_p**2]])
    if Sigma_v == None:
        Correlations_v = np.eye(K) # Correlation structure of the random coefficients
        Std_v = np.ones((K,)) # Standard deviations of the random coefficients
        Sigma_v = (np.eye(K) * Std_v).dot(Correlations_v).dot(np.eye(K) * Std_v)
    shares = np.zeros((J,)) # Shares at price vector p = (p_1, ..., p_J)
    shares1 = np.zeros((J,J)) # Shares at price vector 
                              # p'_j = (p_1, ..., p_(j-1), p_j-dp, p_(j+1), ..., p_J)
                              # are stored at the j-th row of the matrix
    shares2 = np.zeros((J,J)) # Shares at price vector 
                              # p'_j = (p_1, ..., p_(j-1), p_j+dp, p_(j+1), ..., p_J)
                              # are stored at the j-th row of the matrix
    
    # Generating of the (Xi,p) pair
    G = np.random.multivariate_normal([Mean_Xi, Mean_p], Sigma_Xi_p, J)
    Xi = G[ : , 0] # Unobservables vector (Xi)
    p = G[ : , 1] # Price vector (p)
    
    # Generating of observables
    X = np.random.multivariate_normal(Mean_X, Sigma_X, J) # Observables (X)
    C = np.array(beta, ndmin=2).dot(X.T) # Mean utility from observables (X*beta)
    
    # Random choice simulations
    for i in xrange(int(N/parsing_size)):
        # Random coefficient individual specific disturbances (parsing_size-by-K matrix): 
        v = np.random.multivariate_normal(np.zeros((K,)), Sigma_v, parsing_size) 
        # Random utility individual-product specific disturbances (parsing_size-by-J matrix):        
        e = np.random.normal(0, Std_e, (parsing_size,J))
        # Utility individual-product specific disturbance caused by deviation 
        # of random coefficients from their mean (parsing_size-by-J matrix):
        d = v.dot(X.T)
        # Individual-product specific utility given prices vector p, observables X,
        # unobservables Xi, random coefficients (beta+v) and shock e (parsing_size-by-J matrix):
        u = alpha*(-p) + C  + Xi + d + e
        # u => 0-1 matrix T, such that T[i,j]=1 iff individual i's decision is 
        # to buy product j:         
        T = [u.T == np.max(u, axis = 1)]
        T = T[0] * [u.T > 0]
        shares = shares + np.sum(T[0], axis=1) # updating of shares on the parsing step
        # Finite difference derivative approximation:
        dp = 1e-3 # Finite difference step
        u1 = u + alpha*dp # Utility of agent i from good j at price vector 
                          # p'_j = (p_1, ..., p_(j-1), p_j-dp, p_(j+1), ..., p_J)
                          # are stored at the (i,j)-th entry of the matrix
        u2 = u - alpha*dp # Utility of agent i from good j at price vector 
                          # p'_j = (p_1, ..., p_(j-1), p_j+dp, p_(j+1), ..., p_J)
                          # are stored at the (i,j)-th entry of the matrix
        for j in xrange(J):
            # Utility vector of agent i at price vector 
            # p'_j = (p_1, ..., p_(j-1), p_j-dp, p_(j+1), ..., p_J)
            # is stored at the i-th row of the matrix:
            x1 = np.hstack((u[ : , :j], u1[ : , j:j+1], u[ : , j+1: ]))
            # Utility vector of agent i at price vector 
            # p'_j = (p_1, ..., p_(j-1), p_j+dp, p_(j+1), ..., p_J)
            # is stored at the i-th row of the matrix:
            x2 = np.hstack((u[ : , :j], u2[ : , j:j+1], u[ : , j+1: ]))
            # u1 => 0-1 matrix T1, such that T1[i,j]=1 iff individual i's decision is 
            # to buy product j: 
            T1 = [x1.T == np.max(x1, axis = 1)]
            T1 = T1[0] * [x1.T > 0]
            shares1[j, : ] = shares1[j, : ] + np.sum(T1[0], axis=1) # updating of the shares1
            # u2 => 0-1 matrix T2, such that T1[i,j]=1 iff individual i's decision is 
            # to buy product j: 
            T2 = [x2.T == np.max(x2, axis = 1)]
            T2 = T2[0] * [x2.T > 0]
            shares2[j, : ] = shares2[j, : ] + np.sum(T2[0], axis=1) # updating of the shares2
        print(i)
    shares = shares/N # normalization by the market size
    dshares = [(shares2[j, j] - shares1[j, j]) / N / (2*dp) for j in xrange(J)] # finite difference derivative
    MC = p + shares/dshares # Estimation of MC via prices and the expression for mark-ups from FOCs
    Out = np.hstack((np.reshape(p,(-1,1)), np.reshape(shares,(-1,1)), 
                     X, np.reshape(MC,(-1,1)))) # The output matrix
    return Out              


