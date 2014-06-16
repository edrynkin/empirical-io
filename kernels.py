# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 04:18:51 2014

@author: evgenidrynkin
"""

from __future__ import print_function, division

import numpy as np
import functools
import scipy.integrate as si
import matplotlib.pylab as plt
import scipy.stats as spstats
import scipy.optimize as opt
import scipy as sp
import numdifftools as nd

def epanechnikov(data,grid,h):
    data=np.reshape(data,(-1,1))
    n1=np.shape(data)[0]
    n2=np.shape(grid)[0]    
    t=np.zeros((n1,n2))
    for i in xrange(n2):
        a1=1 - ( ( data - grid[i] ) / h )**2
        a2=np.zeros(np.shape(a1))
        a=np.hstack((a1,a2))
        s=np.max(  a, axis=1 )
        t[:,i]= s
    return t