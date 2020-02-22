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

font = {'size'   : 22}
mpl.rcParams['legend.fontsize'] = 30
mpl.rc('font', **font)

N_plot = 5000
fig = plt.figure(figsize=(20,20), dpi=100)
ax = fig.gca(projection='3d')
ax.plot(u[:N_plot,0], u[:N_plot,1], u[:N_plot,2], "b--", label='Lorenz attractor')
ax.legend()
#plt.show()
ax.set_xlabel(r'$X_1$', labelpad=20)
ax.set_ylabel(r'$X_2$', labelpad=20)
ax.set_zlabel(r'$X_3$', labelpad=20)
plt.savefig("./Figures_Projection/Lorenz_Attractor.pdf", dpi=1000, bbox_inches="tight")
plt.close()

for proj in [[1,2],[1,3],[2,3]]:
    fig = plt.figure(figsize=(20,20), dpi=100)
    plt.plot(u[:N_plot,proj[0]-1], u[:N_plot,proj[1]-1], "b--", label='2-D projection of Lorenz attractor')
    plt.legend()
    #plt.show()
    plt.xlabel(r'$X_'+str(proj[0])+'$', labelpad=20, fontsize=40)
    plt.ylabel(r'$X_'+str(proj[1])+'$', labelpad=20, fontsize=40)
    plt.savefig("./Figures_Projection/Lorenz_Attractor_2D_"+str(proj[0])+"_"+str(proj[1])+".pdf", dpi=1000, bbox_inches="tight")
    plt.close()








