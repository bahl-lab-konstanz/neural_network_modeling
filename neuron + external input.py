# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:05:59 2021

@author: kunjal
"""
# This is a rate-based model of a single neuron, which is connected to itself 
# and has an external input signal to be integrated as well.

import numpy as np
import matplotlib.pyplot as plt

tau_r = 0.01
#tau_r_list = np.array([0.001, 0.005,0.02,0.05,0.1])
dt = 0.001

t = np.arange(0, 2.5, dt)
w = [1,1]
#w = np.array([[0.5, 0.5],[0.5, 1], [0.5,2],[1,0.5],[1,1],[1,4]]) #(first column is weight of self-connection, weight of the ext input connection)

#v = np.zeros((1, np.size(t)))
#v[0] = 0

x = np.zeros((1, np.size(t)))
x[0, 0:(np.size(t)//2)] = 1 #np.array((np.arange(np.size(t)//2)/(np.size(t)//2)))
print(x)
x_plt = np.ndarray.flatten(x[0,:])

u = np.concatenate((np.zeros((1, np.size(t))), x), axis=0)
u[0,0] = 0

def sig(a):
    return 1/(1+np.exp(-a))

for i in range(np.size(t)-1):
    u[0, i+1] = u[0,i] + (sig(w @ u[:,i])-u[0,i])*(dt/tau_r)

v = np.ndarray.flatten(u[0,:])

plt.figure(figsize=(8,5.5), dpi=200)
#plt.ylim(top=1.2, bottom=0)
plt.ylabel('Firing Rate')
plt.xlabel('Time (s)')
plt.plot(t, x_plt, color='k', label='External input')
plt.plot(t, v, '-', label= 'Neuron output')
#'w_self ='+str(w[2,0])+', w_ext='+str(w[2,1]))
plt.legend()


'''
# Different values of weights
plt.figure(figsize=(7,5.5), dpi=250)
#plt.ylim(top=1.2, bottom=0)
plt.ylabel('Firing Rate')
plt.xlabel('Time (s)')
plt.plot(t,x_plt, color='k', label='External input')

for j in range(np.shape(w)[0]):
    for i in range(np.size(t)-1):
        u[0, i+1] = u[0,i] + (sig(w[j,:] @ u[:,i])-u[0,i])*(dt/tau_r)
    v = np.ndarray.flatten(u[0,:])
    plt.plot(t, v, '-', label='w_self ='+str(w[j,0])+', w_ext='+str(w[j,1]))
    plt.legend()
'''

'''
#different time constant values
plt.figure(figsize=(7,5.5), dpi=200)
#plt.ylim(top=1.2, bottom=0)
plt.ylabel('Firing Rate')
plt.xlabel('Time (s)')
plt.plot(t,x_plt, color='k', label='External input')

for j in range(np.size(tau_r_list)):
    for i in range(np.size(t)-1):
        u[0, i+1] = u[0,i] + (sig(w[2,:] @ u[:,i])-u[0,i])*(dt/tau_r_list[j])
    v = np.ndarray.flatten(u[0,:])
    plt.plot(t, v, '-', label='neuron tau = '+str(tau_r_list[j]) )
# 'w_self ='+str(w[2,0])+', w_ext='+str(w[2,1]))
    plt.legend()
'''
