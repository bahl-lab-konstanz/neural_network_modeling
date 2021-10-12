# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:07:59 2021

@author: kunjal
"""

#this plots the activities of the integrator for different values of noise 
#added to the input

'''
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#constants
tau_n = 0.01
w = 0.99
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
'''

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#time and initial condition
t = np.arange(0, 3, 0.001)
y0 = [0]

tau_neuron = 0.01

sdev_list = np.arange(0.2,1.0,0.2)
I_val = 2

#self-connectivity values
#w_list = np.arange(0,1,0.1)
w = 0.4

y_array_noise = np.zeros(shape=(np.size(sdev_list),np.size(t)))
fig1, ax1 = plt.subplots(figsize=(15,10),nrows=1,ncols=1,sharex=False,
                           sharey=False,tight_layout=True)
for k in range(0,np.size(sdev_list)): 
    
    def firing_rate_eq(y, t): return ((-y + w*y + I_val+(np.random.normal(loc=0.0, scale=sdev_list[k]))) / tau_neuron)
    y = odeint(firing_rate_eq, y0, t)
    y_array_noise[k,:] = np.array(y).flatten()
    
    ax1.plot(t,y_array_noise[k,:], label='neuron time constant = '+str(sdev_list[k]))

ax1.set_xlabel('time(s)', fontsize=18)
ax1.set_ylabel('firing rate', fontsize=18)
ax1.legend(fontsize=18)
fig1.suptitle('Activity with changing noise std deviation. \n w='+str(w)+', external input = '+str(I_val), fontsize=22)
plt.show()