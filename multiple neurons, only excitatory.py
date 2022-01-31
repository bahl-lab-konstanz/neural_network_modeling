# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:31:31 2021

@author: kunjal
"""

# multiple neurons, all connected to every other neuron with constant weights.
# They all get the same external input signal, to be integrated. The output
# is the average of all of them.

# There are 4 neurons, and a single input to all of them.

import numpy as np
import matplotlib.pyplot as plt

dt = 0.001                  #setting time-step
t = np.arange(0, 1, dt)     #constructing the time matrix
tau_r = 0.01                #setting time constant of the neuron 

input_no = 1       # setting the number of input neurons
u = np.zeros((input_no, np.size(t))) # building the input matrix
halftime = (np.size(t)//2)
u[0, 0:halftime] = 1 #np.array((np.arange(np.size(t)//2)/500)**2) 
                            # there is constant input for the first 
                            #half of the interval, and 0 input for the second half
u_g = np.reshape(u,(np.size(t),))

output_no = 4      #setting the number of output neurons
v = np.zeros((output_no, np.size(t))) #constructing output matrix
# v[:,0] = setting initial conditions for the output neurons

W = np.array([[1],[1],[1],[1]])  # W is of size (no of outputs, no of inputs)

M = np.array([[0, 0.5, 0.5, 0.5],[0.5, 0, 0.5, 0.5],
              [0.5, 0.5, 0, 0.5],[0.5, 0.5, 0.5, 0]])   
                       # M is of size (no of outputs, no of outputs)
                       # here there are no self-connections

def sig(a):
    return 1/(1+np.exp(-a))

for i in range(np.size(t)-1):
    v[:,i+1] = v[:,i] + (sig(W@u[:,i] + M@v[:,i]) - v[:,i]) * (dt/tau_r)  

# plotting the firing rates of the output neurons across time
fig = plt.figure(figsize=(6,5.5), dpi=200)
ax1 = fig.add_subplot(2,2,1)
ax1.plot(t, v[0,:], '-')
ax1.plot(t, u_g)
#ax1.axvline(x=t[halftime], color='k')
ax1.set_ylabel('Firing Rate')
ax1.set_xlabel('Time (s)')
ax1.set_title('neuron 1')

ax2 = fig.add_subplot(2,2,2)
ax2.plot(t, v[1,:], '-', label='Output firing rate')
ax2.plot(t, u_g, label = 'input values')
#ax2.axvline(x=t[halftime], color='k')
ax2.set_xlabel('Time (s)')
ax2.set_title('neuron 2')
ax2.legend(fontsize = 6)

ax3 = fig.add_subplot(2,2,3)
ax3.plot(t, v[2,:])
ax3.plot(t, u_g)
#ax3.axvline(x=t[halftime], color='k')
ax3.set_ylabel('Firing Rate')
ax3.set_xlabel('Time (s)')
ax3.set_title('neuron 3')

ax4 = fig.add_subplot(2,2,4)
ax4.plot(t, v[3,:])
ax4.plot(t, u_g)
#ax4.axvline(x=t[halftime], color='k')
ax4.set_xlabel('Time (s)')
ax4.set_title('neuron 4')

fig.suptitle('4 neurons connected by excitation, Constant input')
plt.tight_layout()
plt.show()