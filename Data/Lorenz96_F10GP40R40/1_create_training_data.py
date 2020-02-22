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

with open(save_path + "/SVD_Data/svd_data.pickle", "rb") as file:
    data = pickle.load(file)
    u = data["u_r"]
    dt = data["dt"]
    del data

N = np.shape(u)[0]

print(u.shape)

N_data_train = int(u.shape[0])//2
N_data_test = int(u.shape[0])//2

[u_train, u_test, _] = np.split(u, [N_data_train, N_data_train+N_data_test], axis=0)
print("Traing data shape: ")
print(u_test.shape)
print(u_train.shape)

train_input_sequence = u_train
test_input_sequence = u_test

dl_max = 20000
pl_max = 20000
max_idx = np.shape(test_input_sequence)[0] - pl_max
min_idx = dl_max
idx = np.arange(min_idx, max_idx)
np.random.shuffle(idx)
testing_ic_indexes = idx

attractor_std = np.std(train_input_sequence, axis=0)
attractor_std = np.array(attractor_std).flatten(-1)
print(np.shape(attractor_std))

print(train_input_sequence.shape)
data = {"train_input_sequence":train_input_sequence,
        "attractor_std":attractor_std, "dt":dt}

with open(save_path + "/Data/training_data_N{:d}.pickle".format(N_data_train), "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

del data

data = {"test_input_sequence":test_input_sequence,
        "testing_ic_indexes":idx,
        "attractor_std":attractor_std, "dt":dt}
print(test_input_sequence.shape)

with open(save_path + "/Data/testing_data_N{:d}.pickle".format(N_data_train), "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
del data






