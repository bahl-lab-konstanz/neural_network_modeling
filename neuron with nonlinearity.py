# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:32:15 2021

@author: kunjal
"""

#This is a rate-based model of a single neuron, the activation function is a 
# nonlinearity (sigmoid function), and there is no other external input

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 1, 0.001)
w= [0.1, 0.5, 0.9, 1.5, 4, 10]

v = np.zeros([np.size(w), np.size(t)])
v[0] = 0
u = v

tau_r = 0.01
dt = 0.001

def sig(a):
    return 1/(1+np.exp(-a))

plt.figure(dpi=200)
plt.xlabel('Firing Rate')
plt.ylabel('Time (s)')

# v[j, i+1] = v[j,i] + (sig(w[j]*u[j,i])-v[j,i])*(dt/tau_r)

for j in range(np.size(w)):
    for i in range(np.size(t)-1):
        v[j, i+1] = v[j,i] + (sig(w[j]*u[j,i])-v[j,i])*(dt/tau_r)
        u[j, i+1] = v[j, i+1]
    plt.plot(t, v[j,:], '-', label='w = '+str(w[j]))
    plt.legend(fontsize=10)

plt.show()