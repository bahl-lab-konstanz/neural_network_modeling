# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 04:04:15 2021

@author: kunjal
"""

import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
t= np.arange(0,3, dt)
y = np.zeros_like(t)
y[0] = 0

tau = 0.01
I = 2*np.ones_like(t)
w = 0.4

rng = np.random.default_rng(5445)

#Equation:
#tau*dr = (-r + w*r + I(t))*dt + I(t)*np.sqrt(dt)*np.random.normal(0,1)

for i in range(np.size(t)):
    if i > 0:
        y[i] = y[i-1] + ((-y[i-1] + w*y[i-1] + I[i])*dt)/tau + (I[i]/(tau*100))*np.sqrt(dt)*rng.normal()

plt.plot(t, y) 