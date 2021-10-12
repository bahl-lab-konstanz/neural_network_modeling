# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:32:48 2021

@author: kunjal
"""

#This gives the activity of the integrator vs time, for both the numerical
#solution and the solution of analytical form fitted to the numerical one.

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#constants and initial values
y0 = [0]
w = 0.8
tau_n = 0.01
I = 1
t = np.arange(0, 3, 0.001)

#function for curvefit
def func_onset(x, A, b, C):
    return A*(1-np.exp(-b*x)) + C
 
def firing_rate_eq(y, t): return ((-y + w*y + I )/ tau_n)
y = odeint(firing_rate_eq, y0, t)
y_array = np.array(y).flatten()

popt, pcov = curve_fit(func_onset,t,y_array)
print(popt)
print('std dev of A =', np.sqrt(pcov[1,1]))

#add a way to reject fits with std > 0.5 for 

tau_int_fitted = (1/popt[1])
tau_int_analytical = tau_n/(1-w)

print('fitted value =', tau_int_fitted)
print('analytical value = ', tau_int_analytical)

fig = plt.figure()
plt.plot(t,y_array, t, func_onset(t, popt[0], popt[1], popt[2]))

