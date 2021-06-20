import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
from math import floor

#%%
#parameters
N = 250 # number of neurons
tau = 10*ms # membrane time constant (in eqs)
r = 0*ms # refractory period
d = 0*ms # delay from presynaptic firing to voltage change in the postsynaptic neuron 
v0 = 2.1e-3 # in eqs
w = 2e-4 # synaptic weight
time = 60 # the time for which the model is run, in milliseconds
time_step = 0.1 # the time step for numerical integration, in milliseconds
prob = 0.4 #connection probability

#set up a 1000 neuron model
start_scope()
defaultclock.dt = time_step*ms

eqs = '''
dv/dt = (v0-v)/tau :1
'''

G = NeuronGroup(N, eqs, threshold = 'v>2e-3', reset='v=0', refractory=r, 
                method='exact')
rng = np.random.default_rng(seed=216)
G.v = rng.random(N)*1.9e-3

S_E = Synapses(G,G, on_pre='v_post+=w')
S_E.connect(condition='i!=j', p=prob)
S_E.delay = d

#statem = StateMonitor(G, 'v', record=True)
spikem = SpikeMonitor(G)
net = Network(G, S_E, spikem)

#run it for a few milliseconds
net.run(time*ms)

#%%
# Calculating firing rates of individual neurons, and the average firing rate of the network
ncols = int((time/time_step)+1) # number of time points, including zero
nrows = N # number of neurons
spike_array = np.zeros([nrows, ncols])
rate_array = np.zeros([nrows, ncols])

T = 2 # the window size, in ms, for calculating firing rate
x = floor(T/time_step) # number of time-points in the interval T
T_1 = ((ncols-1)-(floor((ncols-1)/x)*x)+1)*time_step # for the time period at the end of the sequence of time points 
t_points = np.array((spikem.t/ms)/time_step, dtype=int)

# converting spike data into a matrix with the rows corresponding to individual 
#neurons and the columns corresponding to time points. Time points at which a 
#spike occurs have value 1, others have value 0
for i_1 in range(N):
    for j_1 in range(len(spikem.i)):
        if spikem.i[j_1] == i_1:
            spike_array[i_1, t_points[j_1]] = 1

for i_2 in range(N):
    for j_2 in range(floor((ncols-1)/x)):
        rate_array[i_2,(x*j_2+0):(x*j_2+x)] = sum(spike_array[i_2,(x*j_2+0):(x*j_2+x)])/T   
    rate_array[i_2,(floor((ncols-1)/x))*x:ncols] = sum(spike_array[i_2,((floor((ncols-1)/x))*x):ncols])/T_1 

total_rate = (np.sum(rate_array, axis=0))/N # average firing rate over all neurons in the network

#%%
#plot the spiking and the firing rate over time
fig, ax = plt.subplots(figsize=(25,15), nrows=2, ncols=1, sharex=False,)
fig.suptitle('Number of neurons= '+str(N)+', connection probability= '+str(prob)+
             ', time window for averaging= '+str(T)+' ms', size=20)

ax[0].plot(spikem.t[0:400]/ms, spikem.i[0:400], '.k')
ax[0].set_xlabel('Time (ms)', size = 16)
ax[0].set_ylabel('Neuron index', size = 16)

ax[1].set_xlabel('Time (ms)', size = 16)
ax[1].set_ylabel('Firing rate (spikes/sec)', size = 16) 
#for i in range(N):
#    ax[1].plot(np.arange(0,ncols,1)*time_step, rate_array[i,:], color='grey', linestyle='-')
ax[1].plot(np.arange(0,ncols,1)*time_step, total_rate, '-k')

plt.tight_layout()