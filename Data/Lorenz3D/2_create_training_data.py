#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import pickle

with open("./Simulation_Data/lorenz3D_data.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["u"]
    sigma = data["sigma"]
    beta = data["beta"]
    rho = data["rho"]
    dt = data["dt"]

n = u.shape[0]
T_train_max = n*dt/2
print("Max T_train:")
print(T_train_max)

T_train = 1000.005
T_test = 1000.005

# T_train = 800
# T_test = 200

N_train = int(T_train//dt)
N_test = int(T_test//dt)


dudt = (u[1:]-u[:-1])/dt
u = u[:-1,:]

u_train = u[:N_train, :]
u_test = u[N_train:N_train+N_test, :]

dudt_train = dudt[:N_train, :]
dudt_test = dudt[N_train:N_train+N_test, :]


train_input_sequence = u_train
train_target_sequence = dudt_train
test_input_sequence = u_test
test_target_sequence = dudt_test

print("Number of training samples: {}".format(train_input_sequence.shape))
print("Number of testing samples: {}".format(test_input_sequence.shape))

attractor_std = np.std(train_input_sequence, axis=0)

dl_max = 20000
pl_max = 20000
max_idx = np.shape(test_input_sequence)[0] - pl_max
min_idx = dl_max
idx = np.arange(min_idx, max_idx)
np.random.shuffle(idx)
testing_ic_indexes = idx

print("Shape of initial conditions: {:}".format(testing_ic_indexes.shape))

data = {
"train_input_sequence":train_input_sequence,
"train_target_sequence":train_target_sequence,
"attractor_std":attractor_std,
"dt":dt,"sigma":sigma,
"beta":beta,
"rho":rho,
}

with open("./Data/training_data_N{:d}.pickle".format(N_train), "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

data = {
"test_input_sequence":test_input_sequence,
"test_target_sequence":test_target_sequence,
"attractor_std":attractor_std,
"testing_ic_indexes":testing_ic_indexes,
"dt":dt,"sigma":sigma,
"beta":beta,
"rho":rho,
}

with open("./Data/testing_data_N{:d}.pickle".format(N_test), "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


