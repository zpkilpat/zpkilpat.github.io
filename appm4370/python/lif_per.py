#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lif_perforce.py

simulates and computes statistics of the leaky integrate and fire model.
"""

import numpy as np
import matplotlib.pyplot as plt

taum = 1    # membrane time constant (ms)
urest = 0   # resting potential (mV)
R = 1       # resistance (ohms)
uth = 1     # spiking threshold (mV)
A = 1.5     # current modulation amplitude (mA)

T = 20      # total time to run
dt = 0.001   # time step
nt = int(np.round(T/dt)+1)     # number of entries in vector array
tvec = np.linspace(0,T,nt)

u = np.zeros(nt);   # vector of voltage entries
st = 0              # initialize vector to store spike times

for j in np.arange(nt-1):
    I = A*np.sin(tvec[j])               # 
    u[j+1] = u[j]+dt*(R*I-u[j])/taum    # update the voltage
    if u[j+1]>uth:
        u[j+1]=urest                    # reset the spike voltage
        st = np.append(st,tvec[j+1])    # add spike time to vector
 
print(st)
    
fig = plt.figure(figsize=(5,5))       
plt.plot(tvec,u,linewidth=4.0)
plt.xlabel('time (ms)')
plt.ylabel('voltage (mV)')
plt.show()
fig.savefig('lif_perforce.png', dpi=fig.dpi)