import numpy as np
from numpy import pi
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp2d
from Utils import KS


#------------------------------------------------------------------------------
# define data and initialize simulation
L    = 100/(2*pi)
N    = 512
dt   = 0.25
ninittransients = 10000
tend = 50000 + ninittransients  #50000
dns  = KS.KS(L=L, N=N, dt=dt, tend=tend)


N_data_train = 100000
N_data_test = 100000
dl_max = 20000
pl_max = 20000


#------------------------------------------------------------------------------
# simulate initial transient
dns.simulate()
# convert to physical space
dns.fou2real()


u = dns.uu[ninittransients:]

print(u.shape)

N_train_max = int(u.shape[0])
print('Max N_train:')
print(N_train_max)



[u_train, u_test, _] = np.split(u, [N_data_train, N_data_train+N_data_test], axis=0)
print("Traing data shape: ")
print(u_test.shape)
print(u_train.shape)

train_input_sequence = u_train
test_input_sequence = u_test


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

with open("./Data/training_data_N{:d}.pickle".format(N_data_train), "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

del data

data = {"test_input_sequence":test_input_sequence,
        "testing_ic_indexes":testing_ic_indexes,
        "attractor_std":attractor_std, "dt":dt}
print(test_input_sequence.shape)

with open("./Data/testing_data_N{:d}.pickle".format(N_data_train), "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    
del data



data = {"dns":dns,
        "L":L,
        "N":N, "dt":dt}
print(test_input_sequence.shape)

with open("./Data/simulation_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    
del data





