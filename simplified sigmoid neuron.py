# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 20:23:11 2021

@author: kunjal
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 1, 0.001)
w= [70, 100] #(weight of self-connection, weight of the ext input connection)

v = np.zeros((1, np.size(t)))
#v[0] = 0
x = np.zeros((1, np.size(t)))
x[0, 0:(np.size(x)//2)] = 1

 
u = np.concatenate((np.zeros((1, np.size(t))), x), axis=0)

tau_r = 0.01
dt = 0.001

def sig(a):
    return 1/(1+np.exp(-a))

#for i in range(np.size(t)-1):
#    v[0, i+1] = v[0,i] + (sig(w @ u[:,i])-v[0,i])*(dt/tau_r)
#    if i == 200:
#        print(u[:,200])
    
for i in range(np.size(t)-1):
    if i < 200: 
        I = 0
    elif i >= 200 and i<400:
        I = 1
    else:
        I = 0
    v[0, i+1] = v[0,i] + (sig(I)-v[0,i]) * (dt/tau_r)
    


v = np.ndarray.flatten(v)

plt.figure(dpi=200)
plt.plot(t, v, '-', label='w_self ='+str(w[0])+', w_ext='+str(w[1]))
plt.legend(fontsize=10)
plt.ylabel('Firing Rate')
plt.xlabel('Time (s)')

