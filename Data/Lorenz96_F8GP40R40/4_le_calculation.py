#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
from scipy.integrate import ode
import numpy as np
import sys
import pickle
from Utils.lorenz96 import *

import matplotlib
# Plotting parameters
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm

plt.rcParams["text.usetex"] = True
plt.rcParams['xtick.major.pad']='20'
plt.rcParams['ytick.major.pad']='20'
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)

# F=8
GP=40
RDIM=40
J=40
T=20
dt=0.01
NUM_ICS=10
epsilon=1e-8
N_MED=100

# T = 2
# dt = 0.1

for F in [8, 10]:

    base_path = "../Lorenz96_F{:d}GP{:}".format(F, GP)
    save_path = "../Lorenz96_F{:d}GP{:}R{:}".format(F, GP, RDIM)

    file_name = base_path + "/Simulation_Data/F"+str(F)+"_data.pickle"
    with open(file_name, "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        data = pickle.load(file)
        u = data["X"]
        dt = data["dt"]
        del data

    # data_std = pow(np.std(u,1),2)
    # logdata_std = np.log(data_std)
    # logdata_std = np.mean(logdata_std)
    # print(logdata_std)
    # fix random seed for reproducibility
    np.random.seed(100)


    N=int(T/dt)

    N_ICS = np.shape(u)[0]

    # COMPUTING THE EXPECTED VALUE OF THE DISTANCE BETWEEN TWO POINTS IN THE ATTRACTOR
    expDist=[]
    for i in range(N_MED):
        print("MEAN EXPECTED DISTANCE IC {:}/{:}".format(i, N_MED))
        for j in range(N_MED):
            ici = np.random.randint(0,N_ICS)
            icj = np.random.randint(0,N_ICS)
            dist_=np.linalg.norm(u[ici]-u[icj])
            expDist.append(dist_)
    expDist=np.array(expDist)
    mexpDist=expDist.mean()
    print("MEAN EXPECTED DISTANCE:")
    print(mexpDist)
    # mexpDist=32.46879666896768

    logMexpDist=np.log(mexpDist)

    logabs_error_evol_all=[]
    slopes_all=[]
    for i in range(NUM_ICS):
        print("IC = {:}/{:}".format(i,NUM_ICS))
        ic = np.random.randint(0,N_ICS)
        X0=np.reshape(u[ic], (1,-1))

        print(epsilon)
        size_=np.shape(X0)
        X0_pert=X0+np.random.normal(0, epsilon, size=size_)
        
        logabs_error_evol=[]
        print("\n")
        print("Generate time series\n")
        # Generate time series
        for i in range(N):
            X0 = RK4(Lorenz96,X0,0,dt, F);
            X0_pert = RK4(Lorenz96,X0_pert,0,dt, F);
            abs_error = np.linalg.norm(X0-X0_pert)
            logabs_error_evol.append(np.log(abs_error))

            print("{:d}/{:d}".format(i, N))
            sys.stdout.write("\033[F")


        idx = np.where(logabs_error_evol<0.9*logMexpDist)
        idx = idx[0][-1]
        print(idx)
        Y=logabs_error_evol[:idx]
        X=np.linspace(0, len(Y)*dt, len(Y))
        Y=np.array(Y)
        X=np.array(X)
        slope=((X*Y).mean() - X.mean()*Y.mean()) / ((X**2).mean() - (X.mean())**2)
        # print(ark)

        logabs_error_evol=np.array(logabs_error_evol)
        logabs_error_evol_all.append(logabs_error_evol)
        slopes_all.append(slope)

    logabs_error_evol_all=np.array(logabs_error_evol_all)
    slopes_all=np.array(slopes_all)

    LE=np.mean(slopes_all)

    print("MAXIMUM LYAPUNOV EXPONENT = {:}".format(LE))
    time_axis=np.linspace(0, T, N)

    log_val_error_0=np.mean(logabs_error_evol_all[:,0])
    plt.plot(time_axis, logabs_error_evol_all.T)
    plt.plot(time_axis, np.ones_like(time_axis)*logMexpDist, "--om")
    plt.plot(time_axis, log_val_error_0+time_axis*LE, "--k")
    plt.title("LE={:}".format(LE))
    plt.savefig("./Figures_LE/LYAPUNOV_EXPONENT_N{:}_T{:}_NUMICS{:}_F{:}_RDIM{:}".format(N,T,NUM_ICS, F, RDIM))
    # plt.show()
    plt.close()








