#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import os # for saving
import pickle
import sys
from scipy.linalg import svd
from scipy.linalg import eig
from scipy.linalg import lstsq
from numpy.linalg import norm

F=10
GP=40
RDIM=40
save_path = "../Lorenz96_F{:d}GP{:d}R{:d}".format(F,GP,RDIM)


import matplotlib
# Plotting parameters
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm

plt.rcParams["text.usetex"] = True
plt.rcParams['xtick.major.pad']='20'
plt.rcParams['ytick.major.pad']='20'
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)

with open(save_path + "/SVD_Data/svd_data.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    D = data["D"]
    dt = data["dt"]
    u = data["u"]
    u_r = data["u_r"]
    u_rec = data["u_rec"]
    RDIM = data["RDIM"]
    energy = data["energy"]
    del data



plt.plot(energy)
plt.title("Cummulative energy of Lorenz 96 with F=10")
plt.plot(np.linspace(1,35,len(energy)), np.ones_like(energy)*energy[34], '--r', label=r"98 \%", linewidth=2)
plt.plot(np.ones_like(energy)*35, np.linspace(0,energy[34],len(energy)), '--r', linewidth=2)
plt.xlim([1,40])
plt.ylim([0,1.05])
plt.legend()
# plt.show()
plt.savefig(save_path + "/Figures/energy_F{:}.png".format(F), bbox_inches="tight")
plt.close()

vmin = u.min()
vmax = u.max()

for N_plot in [1000]:
    u_plot = u_r[:N_plot,:]
    N_plot = np.shape(u_plot)[0]
    # Plotting the contour plot
    fig = plt.subplots()
    n, s = np.meshgrid(np.arange(N_plot), np.array(range(RDIM))+1)
    plt.contourf(s, n, np.transpose(u_plot), 100, cmap=plt.get_cmap("seismic"), levels=np.linspace(u_plot.min(), u_plot.max(), 100))
    plt.colorbar()
    plt.xlabel(r"$u$")
    plt.ylabel(r"$n, \quad t=n \cdot {:}$".format(dt))
    plt.savefig(save_path + "/Figures/Plot_UR_F{:}.png".format(F), bbox_inches="tight")
    plt.close()
    u_plot = u_rec[:N_plot,:]
    N_plot = np.shape(u_plot)[0]
    # Plotting the contour plot
    fig = plt.subplots()
    n, s = np.meshgrid(np.arange(N_plot), np.array(range(D))+1)
    plt.contourf(s, n, np.transpose(u_plot), 100, cmap=plt.get_cmap("seismic"), levels=np.linspace(u_plot.min(), u_plot.max(), 100))
    plt.colorbar()
    plt.xlabel(r"$u$")
    plt.ylabel(r"$n, \quad t=n \cdot {:}$".format(dt))
    plt.savefig(save_path + "/Figures/Plot_UR_rec_F{:}.png".format(F), bbox_inches="tight")
    plt.close()
    u_plot = u[:N_plot,:]
    N_plot = np.shape(u_plot)[0]
    # Plotting the contour plot
    fig = plt.subplots()
    n, s = np.meshgrid(np.arange(N_plot), np.array(range(D))+1)
    plt.contourf(s, n, np.transpose(u_plot), 100, cmap=plt.get_cmap("seismic"), levels=np.linspace(u_plot.min(), u_plot.max(), 100))
    plt.colorbar()
    plt.xlabel(r"$u$")
    plt.ylabel(r"$n, \quad t=n \cdot {:}$".format(dt))
    plt.savefig(save_path + "/Figures/Plot_U_F{:}.png".format(F), bbox_inches="tight")
    plt.close()


