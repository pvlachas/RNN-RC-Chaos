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

F=8
GP=40
RDIM=40
base_path = "../Lorenz96_F{:d}GP{:}".format(F, GP)
save_path = "../Lorenz96_F{:d}GP{:}R{:}".format(F, GP, RDIM)

file_name = base_path + "/Simulation_Data/F"+str(F)+"_data.pickle"
with open(file_name, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["X"]
    dt = data["dt"]
    del data

T, D = u.shape
print(T,D)
umean = np.mean(u, axis=0)
u_centered = u - umean
u_scaled = u_centered
COV = np.dot(u_scaled.T, u_scaled)/T

U, s, UT = svd(COV ,compute_uv=True,full_matrices=False, overwrite_a=False, check_finite=True)


s2 = np.power(s,2)
energy = np.cumsum(s2)/np.sum(s2)

# import matplotlib
# # Plotting parameters
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import rc
# from matplotlib  import cm

# plt.plot(energy)
# plt.show()

print("ENERGY OF {:} MOST ENERGETIC MODES: ".format(RDIM))
print(energy[RDIM-1])

# ENERGY OF 12 MOST ENERGETIC MODES:
# 0.9898912681036504

UK = U[:,:RDIM]
u_r = np.dot(u_scaled, UK)

u_centered_rec = np.dot(u_r, UK.T)

print("ERROR:")
print(np.mean(np.square(u_centered-u_centered_rec)))

u_rec = u_centered_rec + umean

print("ERROR:")
print(np.mean(np.square(u-u_rec)))


print(u_r.shape)
data = {
    "RDIM":RDIM,
    "D":D,
    "dt":dt,
    "u_r":u_r,
    "u":u,
    "u_rec":u_rec,
    "energy":energy,
    "UK":UK,
    "s":s,
    "umean":umean,
}

with open(save_path + "/SVD_Data/svd_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


