# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 21:40:29 2021

@author: kunjal
"""

# first take the differential equation and solve it numerically, that is, 
#graph y(t) for various values of input and w
#solve it analytically, that is, find the analytical equation and fit the analytical 
#solution to the numerically calculated values and figure out if the fitted 
#coefficients are the same as the analytical coeffs
#also for time constants

#interpret the fitting coefficients

#add noise to the input I'm not sure tbh 

#dy/dt = (-y + w*y + I) / tau_n

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

t = np.arange(0, 3, 0.001)
tau_n = 0.01

y0 = [0]

def I(t): 
    return 1
    '''
    if t>1 and t<2: 
        return 1
    else:
        return 0
'''

#function for curvefit
def func_onset(x, A, b):
    return A*(1-np.exp(-b*x))

w_list = np.arange(0,1,0.1)
w_list = np.append(np.arange(0,1,0.1), [0.91, 0.93, 0.95, 0.97, 0.98, 0.99, 
                                        0.991, 0.993, 0.995, 1.1, 2.0])
tau_int_analytical = np.zeros_like(w_list)
tau_int_fitted = np.zeros_like(w_list)
y_array = np.zeros(shape=(np.size(w_list),np.size(t)))

for i in range(0,np.size(w_list)): 
    
    def firing_rate_eq(y, t): return ((-y + w_list[i]*y + I(t)) / tau_n)
   
    y = odeint(firing_rate_eq, y0, t)
    y_array[i,:] = np.array(y).flatten()
    
    #carrying out curvefitting    
    popt, pcov = curve_fit(func_onset,t,y_array[i,:], maxfev = 3000)
    print(popt)
    tau_int_fitted[i] = (1/popt[1])
    tau_int_analytical[i] = tau_n/(1-w_list[i])

#z = popt[0] * (1 - np.exp(-(popt[1]) * t))

fig, ax = plt.subplots(figsize=(15,10),nrows=1,ncols=1,sharex=False,
                           sharey=False,tight_layout=True)
ax.scatter(w_list,tau_int_fitted, color='b', label='Fitted time constants')
ax.plot(w_list,tau_int_analytical, color='r', label='Analytical time constants')
ax.set_xlabel('Self-connectivity/w', fontsize=18)
ax.set_ylabel('Integrator time constant (s)', fontsize=18)
ax.legend(fontsize=18)
fig.suptitle('Time constants as functions of w. \n Time constant of neuron ='+str(tau_n)+', external input = 1', fontsize=22)

plt.show()




'''What am i plotting? time constants on y axis, w values on x axis. Time constant
 of neuron, I (input), are mentioned in the title. '''
