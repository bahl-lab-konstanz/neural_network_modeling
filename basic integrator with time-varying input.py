# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:23:54 2021

@author: kunjal
"""

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 1, 0.001)

v = np.zeros([np.size(t)])
I = np.zeros_like(v)
I[0:(np.size(I)//2)] = 0.2 #*np.arange(np.size(I)//2)
#print(I)

w= 1

dt = 0.001
tau_n = 0.01

for i in range(np.size(t)-1):
    v[i+1] = v[i] + (dt/tau_n)*(-v[i] + w * v[i] + I[i])

plt.figure(dpi=150)
plt.plot(t, v, color='blue', label = 'time integral')
print(np.shape(t),np.shape(v))
plt.plot(t, I, '-', color = 'k', label = 'input')
plt.ylabel('Firing Rate')
plt.xlabel('Time (s)')
plt.legend()
plt.show()

'''
y_array_w = np.zeros(shape=(np.size(w_list),np.size(t)))
fig3, ax3 = plt.subplots(figsize=(15,10),nrows=1,ncols=1,sharex=False,
                           sharey=False,tight_layout=True)
for j in range(0,np.size(w_list)):     
    def firing_rate_eq(y, t): return ((-y + w_list[j]*y + I_val) / tau_neuron)
    y = odeint(firing_rate_eq, y0, t)
    y_array_w[j,:] = np.array(y).flatten()
    
    ax3.plot(t,y_array_w[j,:], label='Self-connectivity weight = '+"{:.2f}".format(w_list[j]))

ax3.set_xlabel('time(s)', fontsize=22)
ax3.set_ylabel('firing rate', fontsize=22)
ax3.legend(fontsize=16, loc='upper right')
fig3.suptitle('Activity with changing w (self-connectivity). \n time constant of neuron='+str(tau_neuron)+', external input = '+str(I_val), fontsize=24)
plt.show()
'''