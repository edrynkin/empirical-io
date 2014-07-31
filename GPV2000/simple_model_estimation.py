# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 17:38:47 2014

@author: evgeni
"""
from __future__ import division
import numpy as np
from kernels import density_estimation
from scipy.interpolate import interp1d

def bids_inversion(b, I, G, g):
    return b+1/(I-1)*G(b)/g(b)
    
def estimate_values_density(B, kg, hg, rhog, kf, hf, grid):
    I = np.shape(B)[1]
    L = np.shape(B)[0]
    B = B.ravel()
    G = lambda b: np.array([np.sum(B<=x)/(L*I) for x in b])
    g = lambda b: np.reshape(density_estimation(B, kg, hg, b),(-1,))
    V = bids_inversion(B, I, G, g)
    c = (np.min(B)+rhog*hg/2<=B)*(np.max(B)-rhog*hg/2>=B)
    V = V[c]
    f = density_estimation(V, kf, hf, grid)*np.size(V)/(I*L)
    return f
    
def estimate_strategy(B, kg, hg, grid):
    I = np.shape(B)[1]
    L = np.shape(B)[0]
    B = B.ravel()
    G = lambda b: np.array([np.sum(B<=x)/(L*I) for x in b])
    g = lambda b: np.reshape(density_estimation(B, kg, hg, b),(-1,))
    V = bids_inversion(B, I, G, g)
    points = zip(V, B)
    points = sorted(points, key=lambda point: point[0])
    V, B = zip(*points)
    f = interp1d(V, B, bounds_error=False)
    S = f(grid)
    return S
    
def estimate_full(B, kg, hg, rhog, kf, hf, grid):
    I = np.shape(B)[1]
    L = np.shape(B)[0]
    B = B.ravel()
    G = lambda b: np.array([np.sum(B<=x)/(L*I) for x in b])
    g = lambda b: np.reshape(density_estimation(B, kg, hg, b),(-1,))
    V = bids_inversion(B, I, G, g)
    c = (np.min(B)+rhog*hg/2<=B)*(np.max(B)-rhog*hg/2>=B)
    points = zip(V, B, c)
    points = sorted(points, key=lambda point: point[0])
    V, B, c = zip(*points)
    f = interp1d(V, B, bounds_error=False)
    S = f(grid)
    V = np.array(V)
    V = V[np.array(c)]
    D = density_estimation(V, kf, hf, grid)*np.size(V)/(I*L)
    return D, S