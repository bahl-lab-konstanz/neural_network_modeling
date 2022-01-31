# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 02:23:47 2021

@author: kunjal
"""

# multiple (4) neurons, all connected to every other neuron with constant weights.
# They all get the same external input signal, to be integrated. Their firing 
#rates are the output. 
# There are 4 neurons, and a single input to all of them.

import numpy as np
import matplotlib.pyplot as plt

dt = 0.001                  #setting time-step
t = np.arange(0, 1, dt)     #constructing the time matrix
tau_r = 0.02                #setting time constant of the neuron 

input_no = 1       # setting the number of input neurons
u = np.zeros((input_no, np.size(t))) # building the input matrix
halftime = (np.size(t)//2)
u[0, 0:halftime] = 2
u_g = np.reshape(u,(np.size(t),))

 # there is constant input of magnitude j for the 
                            #first half of the interval, and 0 input for the
                            # second half

output_no = 4      #setting the number of output neurons
v = np.zeros((output_no, np.size(t))) #constructing output matrix
# v[:,0] = setting initial conditions for the output neurons


W = np.array([[1, 2, 3, 4],
              [1, 2, 3, 4],
              [1, 2, 3, 4],
              [1, 2, 3, 4]])
W_l = np.array([[1],[1],[1],[1]])  # W is of size (no of outputs, no of inputs)

M = np.array([[0, 0.5, 0.5, 0.5, 0, 1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3],
             [0.5, 0, 0.5, 0.5, 1, 0, 1, 1, 2, 0, 2, 2, 3, 0, 3, 3],
             [0.5, 0.5, 0, 0.5, 1, 1, 0, 1, 2, 2, 0, 2, 3, 3, 0, 3],
             [0.5, 0.5, 0.5, 0, 1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0]])
#M = np.array([[0, 0.5, 0.5, 0.5],[0.5, 0, 0.5, 0.5],[0.5, 0.5, 0, 0.5],[0.5, 0.5, 0.5, 0]])   
                       # M is of size (no of outputs, no of outputs)
                       # here there are no self-connections

def sig(a):
    return 1/(1+np.exp(-a))
  
fig = plt.figure(figsize=(7,4.5), dpi=200)
plt.axvline(x=t[halftime], color='k')
plt.ylabel('Firing Rate')
plt.xlabel('Time (s)')
plt.title('Neuron time constant = 0.02,\n External input value=2')
for j in range(0,output_no):
    W_in = W[:,j]
    W_f = np.reshape(W_in, np.shape(W_l))
    M_f = M[:,(j*output_no):(j*output_no+output_no)]
    for i in range(np.size(t)-1):
        v[:,i+1] = v[:,i] + (sig(W_f@u[:,i] + M_f@v[:,i]) - v[:,i]) * (dt/tau_r)
    plt.plot(t, v[0,:], '-', label = 'self-connectivity weight='+str(M[1,j*output_no])+', input weight='+str(W_in[0]))
    plt.legend(fontsize=7)
    