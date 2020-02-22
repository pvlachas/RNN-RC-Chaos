#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
from scipy.integrate import ode
import numpy as np
import sys
import pickle
import os

F = 8

base_path = '.'

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


with open(base_path + "/Simulation_Data/F{:}_data.pickle".format(F), "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    X = data["X"]
    J = data["J"]
    dt = data["dt"]
    del data

umax = X.max()
umin = X.min()

print(X.shape)
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)

for t in range(300):
    # Plotting the contour plot
    fig = plt.subplots()
    plt.plot(np.arange(J), X[t+10000])
    plt.xlim([0,J-1])
    plt.ylim([umin,umax])
    plt.xlabel(r"node $i$")
    plt.ylabel(r"$u_i$")
    plt.savefig(base_path + "/Figures_Video/Plot_U_{:04}.png".format(t), bbox_inches="tight")
    plt.close()

bashCommand = "ffmpeg -y -r 30 -f image2 -s 1342x830 -i ./Figures_Video/Plot_U_%04d.png -vcodec libx264 -crf 1  -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' lorenz96_F{:d}_signal.mp4".format(F)
os.system(bashCommand)






