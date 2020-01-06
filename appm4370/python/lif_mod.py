#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lif_mod.py

simulates and computes statistics of the leaky integrate and fire model.
"""

import numpy as np
import matplotlib.pyplot as plt

taum = 1    # membrane time constant (ms)
urest = 0   # resting potential (mV)
R = 1       # resistance (ohms)
I = 5       # input current (mA)
uth = 1     # spiking threshold (mV)

T = 10      # total time to run
dt = 0.001   # time step
nt = int(np.round(T/dt)+1)     # number of entries in vector array (mV)
tvec = np.linspace(0,T,nt)     # time vector (ms)

u = np.zeros(nt)   # vector of voltage entries
st = 0             # initialize vector of spike times

for j in np.arange(nt-1):
    u[j+1] = u[j]+dt*(R*I-u[j])/taum;
    if u[j+1]>uth:
        u[j+1]=urest;                   # reset the voltage to resting potential
        st = np.append(st,tvec[j+1])    # add on another spike time

# estimate the rate        
rateest = st[-1]-st[-2]
 
# plot commands
fig = plt.figure()       
plt.plot(tvec,u,linewidth=4.0)
plt.xlabel('time')
plt.ylabel('voltage')
plt.show()
fig.savefig('lif_model.png', dpi=fig.dpi)
