# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 07:21:03 2021

@author: kunjal
"""
#Also check how changing the I, changing tau_n and changing w alter the final activity
#level of the integrator wrt time

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#time and initial condition
t = np.arange(0, 3, 0.001)
y0 = [0]

#constants
tau_n_list = [0.001,0.005,0.01,0.05,0.1]
tau_neuron = 0.01

#external inputs
I_list = np.arange(0,5,1)
I_val = 2

#self-connectivity values
w_list = np.arange(0,1,0.1)
w = 0.4

y_array_t = np.zeros(shape=(np.size(tau_n_list),np.size(t)))
fig1, ax1 = plt.subplots(figsize=(15,10),nrows=1,ncols=1,sharex=False,
                           sharey=False,tight_layout=True)
for k in range(0,np.size(tau_n_list)): 
    def firing_rate_eq(y, t): return ((-y + w*y + I_val) / tau_n_list[k])
    y = odeint(firing_rate_eq, y0, t)
    y_array_t[k,:] = np.array(y).flatten()
    
    ax1.plot(t,y_array_t[k,:], label='neuron time constant = '+str(tau_n_list[k]))

ax1.set_xlabel('time(s)', fontsize=18)
ax1.set_ylabel('firing rate', fontsize=18)
ax1.legend(fontsize=18)
fig1.suptitle('Activity with changing neuron time constant. \n w='+str(w)+', external input = '+str(I_val), fontsize=22)
plt.show()


y_array_i = np.zeros(shape=(np.size(I_list),np.size(t)))
fig2, ax2 = plt.subplots(figsize=(15,10),nrows=1,ncols=1,sharex=False,
                           sharey=False,tight_layout=True)
for i in range(0, np.size(I_list)):
    def I(t): 
        if t>1 and t<2: 
            return I_list[i]
        else:
            return 0
    def firing_rate_eq(y, t): return ((-y + w*y + I(t)) / tau_neuron)
    y = odeint(firing_rate_eq, y0, t)
    y_array_i[i,:] = np.array(y).flatten()
    
    ax2.plot(t,y_array_t[i,:], label='External Input = '+str(I_list[i]))
    
ax2.set_xlabel('time(s)', fontsize=18)
ax2.set_ylabel('firing rate', fontsize=18)
ax2.legend(fontsize=18)
fig2.suptitle('Activity with changing external input to the integrator.\n w='+str(w)+', neuron time constant = '+str(tau_neuron), fontsize=22)
plt.show()

y_array_w = np.zeros(shape=(np.size(w_list),np.size(t)))
fig3, ax3 = plt.subplots(figsize=(15,10),nrows=1,ncols=1,sharex=False,
                           sharey=False,tight_layout=True)
for j in range(0,np.size(w_list)):     
    def firing_rate_eq(y, t): return ((-y + w_list[j]*y + I_val) / tau_neuron)
    y = odeint(firing_rate_eq, y0, t)
    y_array_w[j,:] = np.array(y).flatten()
    
    ax3.plot(t,y_array_w[j,:], label='External Input = '+str(w_list[j]))

ax3.set_xlabel('time(s)', fontsize=18)
ax3.set_ylabel('firing rate', fontsize=18)
ax3.legend(fontsize=10, loc='upper right')
fig3.suptitle('Activity with changing w (self-connectivity). \n time constant of neuron='+str(tau_neuron)+', external input = '+str(I_val), fontsize=22)
plt.show()

#z = popt[0] * (1 - np.exp(-(popt[1]) * t))
