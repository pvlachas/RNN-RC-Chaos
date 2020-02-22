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

base_path = "."

with open(base_path + "/Data/simulation_data.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["dns"].uu
    dt = data["dns"].dt
    N = data["N"]
    del data


umax = u.max()
umin = u.min()

print(u.shape)
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)

for t in range(1000):
    # Plotting the contour plot
    fig = plt.subplots()
    plt.plot(np.arange(N), u[t+10000])
    plt.xlim([0,N-1])
    plt.ylim([umin,umax])
    plt.xlabel(r"node $i$")
    plt.ylabel(r"$u_i$")
    plt.savefig(base_path + "/Figures_Video/Plot_U_{:04}.png".format(t), bbox_inches="tight")
    plt.close()

bashCommand = "ffmpeg -y -r 30 -f image2 -s 1342x830 -i ./Figures_Video/Plot_U_%04d.png -vcodec libx264 -crf 1  -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' KS_V1_signal.mp4"
os.system(bashCommand)






