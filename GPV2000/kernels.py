# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 21:24:39 2014

@author: evgeni
"""
from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal
from scipy.misc import factorial, factorial2
from scipy.special import poch

def kernel_smoothing(X, y, K, h, grid):
    out = np.empty((0,1)) 
    dv = np.prod(h)
    for x in grid:
        X_h = np.array(x - X) / np.array(h)
        K_h = 1/dv * K(X_h)
        est = np.mean(y*K_h)
        out = np.vstack((out, est))
    return out
    
def density_estimation(X, K, h, grid):
    n = np.shape(X)[0]
    return kernel_smoothing(X, np.ones(n), K, h, grid)
    
def conditional_mean_estimation(X, y, K, h, grid):
    return kernel_smoothing(X, y, K, h, grid)/density_estimation(X, K, h, grid)
  
def generate_kernel(name, *args):
    if name == 'uniform':
        K = lambda x: np.array(np.array(1/2*(np.abs(x)<=1)).T, ndmin=2).prod(0)
    elif name == 'triangular':
        K = lambda x: np.array(np.array((1-np.abs(x))*(np.abs(x)<=1)).T, ndmin=2).prod(0)
    elif name == 'epanechnikov':
        K = lambda x: np.array(np.array(3/4*(1-x**2)*(np.abs(x)<=1)).T, ndmin=2).prod(0)
    elif name == 'normal':
        K = lambda x: np.array(np.array(1/np.sqrt(2*np.pi)*np.exp(-x**2/2)).T, ndmin=2).prod(0)
    elif name == 'mvnormal':
        mu = args[0]
        sigma = args[1]
        s = multivariate_normal(mu, sigma)
        K = lambda x: s.pdf(x)
    elif name == 's-kernel':
        """
        s-kernel of order r
        """
        s = args[0]
        r = args[1]
        k_s = lambda x: np.array(factorial2(2*s+1)/(2**(s+1)*factorial(s))\
                        *(1-x**2)**s*(np.abs(x)<=1))
        K = lambda x: np.array(np.array(hansen_coefficient(s, r, x)*k_s(x)).T, ndmin=2).prod(0)
    else:
        print 'No such kernel found. Try another one.'
        K = ()
    return K
    
def check_kernel_order(K, p, lower = -10, upper = 10, num = 1e6):
    dx = (upper-lower)/num
    X = np.linspace(lower, upper, num)
    k = K(X)
    T = ()
    for i in xrange(p+1):
        t = dx * np.sum(k*X**i)
        T = T + (t,)
    b = np.isclose(T[0], 1, rtol = 0, atol = 1e-5) and\
        np.allclose(T[1:p], np.zeros_like(T[1:p]), rtol = 0, atol = 1e-5) and\
        not np.isclose(T[p], 0, rtol = 0, atol = 1e-5)
    return b
    
def hansen_coefficient(s, r, x):
    a = poch(3/2, r/2-1)*poch(3/2+s, r/2-1)/poch(s+1, r/2-1)
    b = np.zeros_like(x)
    for j in xrange(int(r/2)):
        b = b + (-1)**j*poch(1/2+s+r/2, j)*x**(2*j)/factorial(j)/factorial(r/2-1-j)/poch(3/2, j)
    return a*b
