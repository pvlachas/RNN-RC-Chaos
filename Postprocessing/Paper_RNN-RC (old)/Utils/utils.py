#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import pickle
import glob, os
import numpy as np
import argparse

# ADDING PARENT DIRECTORY TO PATH
import os,sys,inspect

# PLOTTING
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def replaceNaN(data):
    data[np.isnan(data)]=float('Inf')
    return data

def computeNRMSEOnTrajectories(target, prediction, std):
    prediction = replaceNaN(prediction)
    # SQUARE ERROR
    serror = np.square(target-prediction)
    # NORMALIZED SQUARE ERROR
    nserror = serror/np.square(std)
    # MEAN (over-space) NORMALIZED SQUARE ERROR
    mnse = np.mean(nserror, axis=2)
    # ROOT MEAN NORMALIZED SQUARE ERROR
    rmnse = np.sqrt(mnse)
    return mnse

def isDivergentTrajectory(rmnse_traj):
    # NUMBER OF NON DIVERGENT TRAJECTORIES, ARE THOSE THAT DO NOT CROSS RMNSE=3 80% of the total time
    if np.sum(rmnse_traj>3)>0.2*len(rmnse_traj):
        return True
    else:
        return False

def getNumberOfDivergentTrajectories(truths_all, predictions_all, data_std):
    nrmse = computeNRMSEOnTrajectories(truths_all, predictions_all, data_std)
    num_divergent_ics = 0
    num_non_divergent_ics = 0
    for rmnse_traj in nrmse:
        if isDivergentTrajectory(rmnse_traj) or isInvalidTrajectory(rmnse_traj):
            num_divergent_ics+=1
            # plt.plot(rmnse_traj)
            # plt.show()
        else:
            num_non_divergent_ics+=1
    return num_divergent_ics, num_non_divergent_ics

def getPrimeNumbers(howMany, start_num):
    primes=[]
    num=start_num
    while(len(primes)<howMany):
       # prime numbers are greater than 1
        for i in range(2,num):
            if (num % i) == 0:
                break
        else:
            primes.append(num)
        num+=1
    return primes



def computeSpectrumPostProcessing(truths_all, predictions_all, data_std, dt):
    nrmse = computeNRMSEOnTrajectories(truths_all, predictions_all, data_std)
    # Of the form [n_ics, T, n_dim]
    spectrum_db = []
    freq = None
    num_deviating = 0
    num_not_deviating = len(predictions_all)
    for ic in range(np.shape(predictions_all)[0]):
        if ((not isDivergentTrajectory(nrmse[ic])) and (not isInvalidTrajectory(nrmse[ic]))):
            data = np.transpose(predictions_all[ic])
            for d in data:
                freq, s_dbfs = dbfft(d, 1/dt)
                spectrum_db.append(s_dbfs)
                pass
        else:
            num_deviating+=1
            num_not_deviating-=1
            pass
    spectrum_db = np.array(spectrum_db).mean(axis=0)
    return spectrum_db, freq, num_not_deviating, num_deviating
    

def dbfft(x, fs):
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """
    N = len(x)  # Length of input sequence
    if N % 2 != 0:
        x = x[:-1]
        N = len(x)
    x = np.reshape(x, (1,N))
    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)
    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    s_mag = np.abs(sp) * 2 / N
    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag)
    s_dbfs = s_dbfs[0]
    return freq, s_dbfs


def isInvalidTrajectory(data):
    # print(np.shape(data))
    diffs = []
    # print(np.shape(data))

    for tau in [10,20,30,40,50]:
        diff_ = np.linalg.norm(data[-tau]-data[-2*tau])
        diffs.append(diff_)
    diffs = np.array(diffs)
    diffs = np.mean(diffs)
    if diffs < 1e-2:
        # print("FOUND CONVERGED")
        # print(diffs)
        converged = True
    else:
        converged = False

    # if np.any(np.abs(data)>5*data.std()) or np.any(np.isnan(data)) or np.any(np.isinf(data)):
    if converged or np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return True
    else:
        return False


def replaceInvalid(data):
    data[np.isnan(data)]=float(10**8)
    # print(np.sign(data[np.isinf(data)]))
    # print(data[np.isinf(data)])
    data[np.isinf(data)]=float(10**8)*np.sign(data[np.isinf(data)])
    return data

def createContour_(fig, ax, data, title, vmin, vmax, cmap, dt):
    # ax.set_title(repr("\textbf{" + "{:s}".format(title) + "}")[1:-1])
    ax.set_title(title)
    t, s = np.meshgrid(np.arange(data.shape[0])*dt, np.arange(data.shape[1]))
    mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both")
    ax.set_rasterization_zorder(-10)
    # fig.colorbar(mp, ax=ax)
    # ax.set_xlabel(r"$Gridpoint$")
    ax.set_xticks([np.arange(data.shape[1]).min(), np.arange(data.shape[1]).max()])
    return mp

def getMeanTrainingTimeOfParallelModel(train_path, NPG, GS, GIL, TYPE):
    # train_file_ = train_results_path + model_name_key + "/data.pickle"
    data_str = (TYPE=="esn")*"/data_member_0.pickle" + (TYPE!="esn")*"/data_0.pickle"
    train_file_ = train_path + data_str
    time_vec = []
    for i in range(int(NPG)):
        train_result = pickle.load(open(train_file_, "rb" ))
        time = train_result["total_training_time"]
        time_vec+=list([time])
    time_vec=np.array(time_vec)
    mean_training_time=np.mean(time_vec)
    # min_training_time=np.min(time_vec)
    # max_training_time=np.max(time_vec)
    return mean_training_time

def getMemoryOfParallelModel(train_path, NPG, GS, GIL, TYPE):
    # train_file_ = train_results_path + model_name_key + "/data.pickle"
    data_str = (TYPE=="esn")*"/data_member_0.pickle" + (TYPE!="esn")*"/data_0.pickle"
    train_file_ = train_path + data_str
    memory_vec = []
    for i in range(int(NPG)):
        train_result = pickle.load(open(train_file_, "rb" ))
        memory = train_result["memory"]
        memory_vec+=list([memory])
    memory_vec=np.array(memory_vec)
    memory_sum=np.sum(memory_vec)
    # memory_max=np.max(memory_vec)
    # print(memory_max)
    # memory_sum=len(memory_vec)*memory_max
    return memory_sum

def getStandardDevOfParallelModel(train_path, NPG, GS, GIL, TYPE):
    # train_file_ = train_results_path + model_name_key + "/data.pickle"
    data_str = (TYPE=="esn")*"/data_member_0.pickle" + (TYPE!="esn")*"/data_0.pickle"
    train_file_ = train_path + data_str
    std_vec = []
    for i in range(int(NPG)):
        train_result = pickle.load(open(train_file_, "rb" ))
        scaler = train_result["scaler"]
        std = list(scaler.data_std)
        std = std[int(GIL):-int(GIL)]
        std_vec+=std
    std_vec=np.array(std_vec)
    return std_vec

def sortModelList(modellist, COLUMN_TO_SORT):
    modellist_array = np.array(modellist)
    list_ = modellist_array[:,COLUMN_TO_SORT]
    # print(list_)
    list_ = np.array([float(x) for x in list_])
    # print(list_)
    idx=list_.argsort()[::-1]
    # print(idx)
    modellist_sorted = modellist_array[idx]
    # print(modellist_sorted)
    return modellist_sorted


















def parseLineDict(line, filename, modeldict):
    temp = line[:-1].split(":")
    # print(temp)
    # MODEL NAME
    model={str(temp[0]):str(temp[1])}
    for i in range(2, len(temp), 2):
        model[str(temp[i])] = float(temp[i+1])
    modeldict[str(temp[1])]=model
    return modeldict

def getAllModelsTestList(saving_path):
    os.chdir(saving_path)
    filename='./test.txt'
    modellist = []
    with open(filename, 'r') as file_object:  
        for line in file_object:
            # print(line)
            modellist=parseLineToList(line, filename, modellist)
    return modellist

def getAllModelsTrainList(saving_path):
    os.chdir(saving_path)
    filename='./test.txt'
    modellist = []
    with open(filename, 'r') as file_object:  
        for line in file_object:
            # print(line)
            modellist=parseLineToList(line, filename, modellist)
    return modellist

def parseLineToList(line, filename, modellist):
    temp = line[:-1].split(":")
    # print(temp)
    if filename == "./test.txt":
        model_name = temp[1]
        num_test_ICS = int(float(temp[3]))
        num_accurate_pred_005_avg_TEST = float(temp[5])
        num_accurate_pred_050_avg_TEST = float(temp[7])
        num_accurate_pred_005_avg_TRAIN = float(temp[9])
        num_accurate_pred_050_avg_TRAIN = float(temp[11])
        error_freq_TRAIN = float(temp[13])
        error_freq_TEST = float(temp[15])
        model=[model_name, num_accurate_pred_005_avg_TEST, num_accurate_pred_050_avg_TEST, num_accurate_pred_005_avg_TRAIN, num_accurate_pred_050_avg_TRAIN, error_freq_TRAIN, error_freq_TEST]
    elif filename == "./train.txt":
        model_name = temp[1]
        memory = float(temp[3])
        total_training_time = float(temp[5])
        n_model_parameters = int(temp[7])
        n_trainable_parameters = int(temp[9])
        model=[model_name, memory, total_training_time, n_model_parameters, n_trainable_parameters]
    else:
        raise ValueError("I do not know how to parse line for filename {:}.".format(filename))
    modellist.append(model)
    return modellist
    
def getUpperLine(hull, data):
    X = []
    Y = []
    vertices = []
    for vert in hull.vertices:
        X.append(data[vert,0])
        Y.append(data[vert,1])
        vertices.append(vert)

    X = np.reshape(X, (-1,1))
    Y = np.reshape(Y, (-1,1))
    # upper_line = np.concatenate((X,Y), axis=1)

    # # Start from the point with maximum x-value
    upper_line = []
    x_max = np.max(X)
    i = np.where(X == x_max)[0]
    # print(i)

    y_candidate = Y[i]
    y_max = np.max(y_candidate)
    j = np.where(y_candidate == y_max)[0]
    # print(j)
    i = i[j][0]
    # print(i)

    all_lines = []
    for j in range(len(vertices)):
        all_lines.append([X[j], Y[j]])

    vert = vertices[i]
    X_ = X[i]
    Y_ = Y[i]
    # print(vertices)
    # print(X)
    # print(Y)
    # print(i)
    upper_line.append([X_, Y_])
    ready = False
    while True:
        i = i + 1
        if i < np.shape(vertices)[0]:
            vert = vertices[i]
        else:
            vert = vertices[i-np.shape(vertices)[0]]
        X_n = data[vert,0]
        Y_n = data[vert,1]
        # if X_n < X_:
        if X_n < X_:
            if Y_n < 1.0 * Y_:
                upper_line.append([X_n, Y_n])
                X_ = X_n
                Y_ = Y_n
            else:
                upper_line[-1] = [X_n, Y_n]
                X_ = X_n
                Y_ = Y_n

        else:
            break
    # print(ARK)
    upper_line = np.array(upper_line)
    all_lines = np.array(all_lines)
    # print(np.shape(upper_line))
    return upper_line, all_lines

