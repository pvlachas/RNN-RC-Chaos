#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
from __future__ import print_function
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
import os
import sys
import argparse


with open("./Data/lorenz_data.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["u"]
    sigma = data["sigma"]
    beta = data["beta"]
    rho = data["rho"]
    del data

font = {'size'   : 22}
mpl.rcParams['legend.fontsize'] = 30
mpl.rc('font', **font)

N_plot = 1000
proj = 0
data_vector = np.reshape(u[:N_plot,proj], (-1,1))
data_vector = data_vector[::2,:]

fig = plt.figure(figsize=(20,10), dpi=100)
plt.plot(np.arange(np.shape(data_vector)[0]), data_vector, "bx", label='time series')
plt.legend()
#plt.show()
plt.xlabel(r'$t$', labelpad=20, fontsize=40)
plt.ylabel(r'$X$', labelpad=20, fontsize=40)
plt.savefig("./Figures_Sequence/Lorenz_sequence_proj_"+str(proj)+"_N_"+str(N_plot)+".pdf", dpi=1000, bbox_inches="tight")
plt.close()



N_plot = 20
proj = 0
data_vector = np.reshape(u[:N_plot,proj], (-1,1))
data_vector = data_vector[::2,:]

fig = plt.figure(figsize=(26,10), dpi=100)
plt.plot(np.arange(np.shape(data_vector)[0]), data_vector, "bx", label='time series', markersize=4, mew=20)
plt.legend()
#plt.show()
plt.xlabel(r'$t$', labelpad=20, fontsize=40)
plt.ylabel(r'$X$', labelpad=20, fontsize=40)
plt.savefig("./Figures_Sequence/Lorenz_sequence_proj_"+str(proj)+"_N_"+str(N_plot)+".pdf", dpi=1000, bbox_inches="tight")
plt.close()





