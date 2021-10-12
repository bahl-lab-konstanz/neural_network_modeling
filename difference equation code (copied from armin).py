# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 18:56:04 2021

@author: kunjal
"""

import numpy as np
import pylab as pl

dt = 0.01
tau_n = 0.5


def my_function(I, r, w):

    # tau*dr/dt = -r + w*r + I(t)
    # dr = (-r + w*r + I(t))*dt/tau
    # r(t+1) = r(t) + dr

    for i in range(len(I) - 1):

        dr = (-r[i] + w*r[i] + I[i])*dt/tau_n

        r[i + 1] = r[i] + dr


for w in [0, 0.1, 0.5]:

    I = np.zeros(int(100/dt))
    I[int(20/dt):int(60/dt)] = 1
    r = np.zeros_like(I)

    my_function(I, r, w)

    pl.plot(r)

pl.show()


#Functions for onset:
def func_onset(x, A, b):
    return A * (1 - np.exp(-b * x))

def func_offset(x, A, b):
    return A * np.exp(-b * x) ####!! C??

#curvefit function from scipy.optimize import curve_fit
popt, pcov = curve_fit(func_onset, time_fit, 
                       np.nanmean(region_mean_preferred, axis=0)
                       [int(20 / 0.5):int(60 / 0.5)])

tau_onset = 1 / popt[1]

