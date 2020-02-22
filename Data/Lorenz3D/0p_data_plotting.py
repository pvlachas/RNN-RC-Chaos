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


mpl.rcParams['legend.fontsize'] = 10


N_plot = 20000
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(u[:N_plot,0], u[:N_plot,1], u[:N_plot,2], "b--", label='Lorenz attractor')
ax.legend()
#plt.show()
ax.set_xlabel(r'$X_1$')
ax.set_ylabel(r'$X_2$')
ax.set_zlabel(r'$X_3$')
plt.savefig("./Figures/Trajectories_plot_{:d}.pdf".format(N_plot), dpi=1000, bbox_inches="tight")
plt.close()



with open("./Data/lorenz_data_ic.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u_IC = data["u_IC"]







