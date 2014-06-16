# -*- coding: utf-8 -*-
"""
Created on Thu May  1 13:42:36 2014

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
import firms as firms

class market(object):
    def __init__(self,demand,firms,type_of_competition,order=None):
        self.demand=demand
        self.firms=firms
        self.comp=type_of_competition
        self.order=order
        
    def solve(self,init_guess,full=False):
        n=self.firms.number_of_firms
        firms=self.firms
        costs=firms.costs
        
        if self.comp == 'Cournot':
            def residual_demand(q_tilde,q):
                r=np.zeros((n,1))
                for i in xrange(n):
                    conj_q = [q[j] for j in xrange(n) if j != i]
                    qq=np.zeros((n,1))
                    for j in xrange(i):
                        qq[j]=conj_q[j]
                    qq[i]=q_tilde[i]
                    for j in xrange(i+1,n):
                        qq[j]=conj_q[j-1]
                    r[i]=self.demand(qq)
                return r
            def __focs__(q):
                f=np.zeros((n,))
                h=1e-6
                for i in xrange(n):
                    q1=np.array(q)
                    q1[i]=q[i]+h
                    dTR=q1[i]*residual_demand(q1,q)[i]-q[i]*residual_demand(q,q)[i]
                    MR=dTR/h
                    MC=(costs[i](q1)-costs[i](q))/h
                    f[i]=MR-MC
                return f
            residual_demand(init_guess,init_guess)
            q_star=opt.fsolve(__focs__, init_guess)
            self.equilibrium_output=q_star
            c=[costs[i](q_star) for i in xrange(n)]
            self.equilibrium_profit=np.reshape(residual_demand(q_star,q_star),(n,))*q_star - c
            self.equilibrium_price=self.demand(q_star)
            if full == True:
                c=[costs[i](q_star) for i in xrange(n)]
                self.equilibrium_profit=np.reshape(residual_demand(q_star,q_star),(n,))*q_star - c
                self.equilibrium_price=self.demand(q_star)
                num=1e6
                grid=np.linspace(0,sum(q_star),num)
                self.equilibrium_cs=sum([self.demand(x) for x in grid])/num
            
        
#        if self.comp == 'Stackelberg':
        
        
        if self.comp == 'Monopoly':
            def profit(q):
                return -q*self.demand(q)-costs(q)
            q_star=opt.fmin(profit,init_guess)
            self.equilibrium_output=q_star
            self.equilibrium_profit=0-profit(q_star)
            self.equilibrium_price=self.demand(q_star)
            if full == True:
                num=1e6
                grid=np.linspace(0,sum(q_star),num)
                self.equilibrium_cs=sum([self.demand(x) for x in grid])/num
                

def free_entry(demand,costs,init_guess):
    n=0
    def c(q):
        return costs(q[0])
    cc=[c]
    f=firms.firms(1,costs)
    m=market(demand,f,'Monopoly')
    m.solve(init_guess)
    profit=m.equilibrium_profit
    while profit>=0:
        n=n+1
        def c(q):
            return costs(q[n])
        cc.append(c)
        f=firms.firms(n+1,cc)
        m=market(demand,f,'Cournot')
        m.solve((n+1)*[init_guess/(n+1)])
        profit=m.equilibrium_profit[0]
    return n
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    