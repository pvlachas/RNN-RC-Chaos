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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm

plt.rcParams["text.usetex"] = True
plt.rcParams['xtick.major.pad']='20'
plt.rcParams['ytick.major.pad']='20'


with open(save_path + "/SVD_Data/svd_data.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u_all = data["u_r"]
    dt = data["dt"]
    del data


for RDIM in [10, 20, 30, 40]:
    u = u_all[:,:RDIM]
    umax = u.max()
    umin = u.min()

    font = {'weight':'normal', 'size':16}
    plt.rc('font', **font)

    os.makedirs(save_path + "/Figures_Video_RDIM_{:}".format(RDIM), exist_ok=True)
    for t in range(300):
        # Plotting the contour plot
        fig = plt.subplots()
        plt.plot(np.arange(RDIM), u[t+10000])
        plt.xlim([0,RDIM-1])
        plt.ylim([umin,umax])
        plt.xlabel(r"PCA mode $k$")
        plt.ylabel(r"$z_k$")
        plt.savefig(save_path + "/Figures_Video_RDIM_{:}/Plot_UR_{:04}.png".format(RDIM, t), bbox_inches="tight")
        plt.close()

    bashCommand = "ffmpeg -y -r 30 -f image2 -s 1342x830 -i ./Figures_Video_RDIM_{:}/Plot_UR_%04d.png -vcodec libx264 -crf 1  -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' lorenz96_F{:d}GP{:d}R_svd_{:d}_signal.mp4".format(RDIM, F, GP, RDIM)
    os.system(bashCommand)

## COMMAND FOR VIDEO
# ffmpeg -r 30 -f image2 -s 1342x830 -i ./Figures_Video/Plot_UR_%04d.png -vcodec libx264 -crf 1  -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" lorenz96_F8R_svd_50_signal.mp4



