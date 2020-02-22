#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
# from __future__ import print_function
import numpy as np
import matplotlib
# Plotting parameters
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm

plt.rcParams["text.usetex"] = True
plt.rcParams['xtick.major.pad']='20'
plt.rcParams['ytick.major.pad']='20'
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)

import pickle

base_path = "."

F=8

with open(base_path + "/Simulation_Data/F"+str(F)+"_data.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    X = data["X"]
    J = data["J"]
    dt = data["dt"]
    del data


for N_plot in [1000, 10000, 100000]:
    x_plot = X[:N_plot,:]
    N_plot = np.shape(x_plot)[0]
    # Plotting the contour plot
    fig = plt.subplots()
    # t, s = np.meshgrid(np.arange(N_plot)*dt, np.array(range(N))+1)
    n, s = np.meshgrid(np.arange(N_plot), np.array(range(J))+1)
    plt.contourf(s, n, np.transpose(x_plot), 15, cmap=plt.get_cmap("seismic"))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$n, \quad t=n \cdot {:}$".format(dt))
    plt.savefig(base_path + "/Figures/Plot_U_first_N{:d}.png".format(N_plot), bbox_inches="tight")
    plt.close()

for N_plot in [1000, 10000, 100000]:
    x_plot = X[-N_plot:,:]
    N_plot = np.shape(x_plot)[0]
    # Plotting the contour plot
    fig = plt.subplots()
    # t, s = np.meshgrid(np.arange(N_plot)*dt, np.array(range(N))+1)
    n, s = np.meshgrid(np.arange(N_plot), np.array(range(J))+1)
    plt.contourf(s, n, np.transpose(x_plot), 15, cmap=plt.get_cmap("seismic"))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$n, \quad t=n \cdot {:}$".format(dt))
    plt.savefig(base_path + "/Figures/Plot_U_last_N{:d}.png".format(N_plot), bbox_inches="tight")
    plt.close()

