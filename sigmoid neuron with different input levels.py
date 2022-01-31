# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 22:06:05 2021

@author: kunjal
"""

'''
# different external input values

import numpy as np
import matplotlib.pyplot as plt

tau_r = 0.01
#tau_r_list = np.array([0.001, 0.005,0.02,0.05,0.1])
dt = 0.001

t = np.arange(0, 0.5, dt)
w = np.array([0, 0])
w_list = np.array([[0.5, 0.5],[0.5, 1], [0.5,2],[1,0.5],[1,1],[1,4]]) #(first column is weight of self-connection, weight of the ext input connection)

#v = np.zeros((1, np.size(t)))
#v[0] = 0
x = np.zeros((1, np.size(t)))

def sig(a):
    return 1/(1+np.exp(-a))

#different time constant values
plt.figure(figsize=(7,5.5), dpi=200)
#plt.ylim(top=1.2, bottom=0)
plt.ylabel('Firing Rate')
plt.xlabel('Time (s)')

for j in range(1,100,20):
    x[0, 0:(np.size(x)//2)] = j
    u = np.concatenate((np.zeros((1, np.size(t))), x), axis=0)
    u[0,0] = 0
    for i in range(np.size(t)-1):
        u[0, i+1] = u[0,i] + (sig(w @ u[:,i])-u[0,i])*(dt/tau_r)
    v = np.ndarray.flatten(u[0,:])
    plt.plot(t, v, '-', label='external input = '+str(j) )
    x_plt = np.ndarray.flatten(x[0,:])
    #plt.plot(t, x_plt, color='k', label='External input')
    plt.legend()
    
# 'w_self ='+str(w[2,0])+', w_ext='+str(w[2,1]))
'''

#restart

import numpy as np
import matplotlib.pyplot as plt

tau_r = 0.01
dt = 0.001
t = np.arange(0, 0.5, dt)
w = np.array([1, 1])

def sig(a):
    return 1/(1+np.exp(-a))

x = np.zeros((1, np.size(t)))

fig = plt.figure(figsize=(7,5.5), dpi=200)
plt.ylabel('Firing Rate')
plt.xlabel('Time (s)')

for j in [1,2,3,10]:
    x[0, 0:(np.size(x)//2)] = j
    u = np.concatenate((np.zeros((1, np.size(t))), x), axis=0)
    u[0,0] = 0
    
    for i in range(np.size(t)-1):
        u[0, i+1] = u[0,i] + (sig(w @ u[:,i])-u[0,i])*(dt/tau_r)
        
    v = np.ndarray.flatten(u[0,:])
    #x_plot = np.ndarray.flatten(x[0,:])
    
    #plt.ylim(top=1.2, bottom=0)
    plt.plot(t, v, '-', label='external input = '+str(j) )
    #plt.plot(t, x_plot, color='k', label='External input')
    plt.legend()
