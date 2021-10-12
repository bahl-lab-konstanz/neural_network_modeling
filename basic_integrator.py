# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 20:13:34 2021

@author: kunjal
"""

#This solves the integrator numerically and plots its activity against time

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#constants
tau_n = 0.01
w = 0.9
I = 1

# function that returns dy/dt
def model(y,t):
    dydt = (-y + w*y + I) / tau_n
    return dydt

# initial condition
y0 = [0]

# time points
t = np.linspace(0,3)

# solve ODE
y = odeint(model,y0,t)

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()

#integrator time constant
tau_i = tau_n/(1-w)
print(tau_i)