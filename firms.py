# -*- coding: utf-8 -*-
"""
Created on Thu May  1 15:41:06 2014

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
import kernels as kernels

class firms(object):
    def __init__(self,number_of_firms,costs):
        self.number_of_firms=number_of_firms
        self.costs=costs
