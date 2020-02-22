#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by:  Jaideep Pathak, University of Maryland
                Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import pickle
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
# from scipy.linalg import lstsq as scipylstsq
# from numpy.linalg import lstsq as numpylstsq
from utils import *
import os
from plotting_utils import *
from global_utils import *
import pickle
import time
import tensorflow as tf

# MEMORY TRACKING
import psutil

from functools import partial
print = partial(print, flush=True)


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class esn_parallel(object):
    def delete(self):
        return 0

    def __init__(self, params):

        if params["RDIM"] % params["num_parallel_groups"]: raise ValueError("ERROR: The num_parallel_groups should divide RDIM.")
        if size != params["num_parallel_groups"]: raise ValueError("ERROR: The num_parallel_groups is not equal to the number or ranks. Aborting...")

        self.display_output = params["display_output"]
        # print("RANDOM SEED: {:}".format(params["worker_id"]))
        # np.random.seed(params["worker_id"])
        self.GPU = True if (tf.test.is_gpu_available()==True) else False

        self.saving_path = params["saving_path"]
        self.model_dir = params["model_dir"]
        self.fig_dir = params["fig_dir"]
        self.results_dir = params["results_dir"]
        self.logfile_dir = params["logfile_dir"]
        self.write_to_log = params["write_to_log"]

        self.main_train_data_path = params["train_data_path"]
        self.main_test_data_path = params["test_data_path"]

        # PARALLEL MODEL
        self.num_parallel_groups = params["num_parallel_groups"]
        self.RDIM = params["RDIM"]
        self.N_used = params["N_used"]
        self.parallel_group_interaction_length = params["parallel_group_interaction_length"]
        params["parallel_group_size"] = int(params["RDIM"]/params["num_parallel_groups"])
        self.parallel_group_size = params["parallel_group_size"]

        self.worker_id = rank
        self.parallel_group_num = rank

        self.RDIM = params["RDIM"]
        self.input_dim = params["parallel_group_size"] + params["parallel_group_interaction_length"] * 2
        self.output_dim = params["parallel_group_size"]

        self.approx_reservoir_size = params["approx_reservoir_size"]
        self.degree = params["degree"]
        self.radius = params["radius"]
        self.sigma_input = params["sigma_input"]
        self.dynamics_length = params["dynamics_length"]
        self.iterative_prediction_length = params["iterative_prediction_length"]
        self.num_test_ICS = params["num_test_ICS"]

        self.regularization = params["regularization"]
        self.scaler_tt = params["scaler"]
        self.scaler = scaler(self.scaler_tt)
        self.noise_level = params["noise_level"]
        self.model_name = self.createModelName(params)

        # MASTER RANK CREATING DIRECTORIES
        if rank==0:
            os.makedirs(self.saving_path + self.model_dir + self.model_name, exist_ok=True)
            os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
            os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
            os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)


    def createMainModelName(self, params):
        keys = self.getKeysInModelName()
        str_ = "GPU-" * self.GPU + "RNN-esn-PARALLEL"
        for key in keys:
            str_ += "-" + keys[key] + "_{:}".format(params[key])
        return str_

    def getKeysInModelName(self):
        keys = {
        'num_parallel_groups':'NUM_PARALL_GROUPS', 
        'RDIM':'RDIM', 
        'N_used':'N_used', 
        'approx_reservoir_size':'SIZE', 
        'degree':'D', 
        'radius':'RADIUS',
        'sigma_input':'SIGMA',
        'dynamics_length':'DL',
        'noise_level':'NL',
        'iterative_prediction_length':'IPL',
        'regularization':'REG',
        #'num_test_ICS':'NICS',
        'num_parallel_groups':'NG', 
        'parallel_group_size':'GS', 
        'parallel_group_interaction_length':'GIL', 
        # 'worker_id':'WID', 
        }
        return keys

    def createModelName(self, params):
        keys = self.getKeysInModelName()
        str_ = "GPU-" * self.GPU + "RNN-esn-PARALLEL"
        for key in keys:
            str_ += "-" + keys[key] + "_{:}".format(params[key])
        return str_

    def getSparseWeights(self, sizex, sizey, radius, sparsity, worker_id=1):
        # W = np.zeros((sizex, sizey))
        # Sparse matrix with elements between 0 and 1
        if self.parallel_group_num==0: print("WEIGHT INIT")
        W = sparse.random(sizex, sizey, density=sparsity, random_state=worker_id)
        # W = sparse.random(sizex, sizey, density=sparsity, random_state=worker_id, data_rvs=np.random.randn)
        # Sparse matrix with elements between -1 and 1
        # W.data *=2
        # W.data -=1 
        # to print the values do W.A
        if self.parallel_group_num==0: print("EIGENVALUE DECOMPOSITION")
        eigenvalues, eigvectors = splinalg.eigs(W)
        eigenvalues = np.abs(eigenvalues)
        W = (W/np.max(eigenvalues))*radius
        return W

    def augmentHidden(self, h):
        h_aug = h.copy()
        # h_aug = pow(h_aug, 2.0)
        # h_aug = np.concatenate((h,h_aug), axis=0)
        h_aug[::2]=pow(h_aug[::2],2.0)
        return h_aug
    def getAugmentedStateSize(self):
        return self.reservoir_size

    # def augmentHidden(self, h):
    #     h_aug = h.copy()
    #     h_aug = pow(h_aug, 2.0)
    #     h_aug = np.concatenate((h,h_aug), axis=0)
    #     return h_aug
    # def getAugmentedStateSize(self):
    #     return 2*self.reservoir_size


    def train(self):
        self.start_time = time.time()

        self.worker_train_data_path, self.worker_test_data_path = createParallelTrainingData(self)
        dynamics_length = self.dynamics_length
        input_dim = self.input_dim
        N_used = self.N_used
        with open(self.worker_train_data_path, "rb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            data = pickle.load(file)
            train_input_sequence = data["train_input_sequence"]
            print("Adding noise to the training data. {:} per mille ".format(self.noise_level))
            train_input_sequence = addNoise(train_input_sequence, self.noise_level)            
            N_all, dim = np.shape(train_input_sequence)
            if self.input_dim > dim: raise ValueError("Requested input dimension is wrong.")
            train_input_sequence = train_input_sequence[:N_used, :input_dim]
            dt = data["dt"]
            del data

        print("##Using {:}/{:} dimensions and {:}/{:} samples ##".format(input_dim, dim, N_used, N_all))


        if self.parallel_group_num==0: print("SCALING")
        train_input_sequence = self.scaler.scaleData(train_input_sequence)

        N, input_dim = np.shape(train_input_sequence)

        # Setting the reservoir size automatically to avoid overfitting
        if self.parallel_group_num==0: print("Initializing the reservoir weights...")
        nodes_per_input = int(np.ceil(self.approx_reservoir_size/input_dim))
        self.reservoir_size = int(input_dim*nodes_per_input)

        self.sparsity = self.degree/self.reservoir_size;

        if self.parallel_group_num==0: print("Computing sparse hidden to hidden weight matrix...")
        W_h = self.getSparseWeights(self.reservoir_size, self.reservoir_size, self.radius, self.sparsity, self.worker_id)

        # Initializing the input weights
        if self.parallel_group_num==0: print("Initializing the input weights...")

        W_in = np.zeros((self.reservoir_size, input_dim))
        q = int(self.reservoir_size/input_dim)
        for i in range(0, input_dim):
            W_in[i*q:(i+1)*q,i] = self.sigma_input * (-1 + 2*np.random.rand(q))

        # TRAINING LENGTH
        tl = N - dynamics_length
        if self.parallel_group_num==0: print("TRAINING: Dynamics prerun...")
        # H_dyn = np.zeros((dynamics_length, 2*self.reservoir_size, 1))
        h = np.zeros((self.reservoir_size, 1))
        for t in range(dynamics_length):
            if self.display_output == True and self.parallel_group_num == 0:
                print("TRAINING - Dynamics prerun: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
            i = np.reshape(train_input_sequence[t], (-1,1))
            h = np.tanh(W_h @ h + W_in @ i)
            # H_dyn[t] = self.augmentHidden(h)
        if self.parallel_group_num==0: print("\n")

        # # FOR PLOTTING THE DYNAMICS
        # dyn_plot_max = 100
        # H_dyn_plot = H_dyn[:,:dyn_plot_max,0]
        # fig_path = self.saving_path + self.fig_dir + self.model_name + "/H_dyn_prerun_plot_{:d}.png".format(self.parallel_group_num)
        # plt.plot(H_dyn_plot)
        # plt.title('Dynamics prerun')
        # plt.savefig(fig_path)
        # plt.close()

        NORMEVERY = 10
        HTH = np.zeros((self.getAugmentedStateSize(), self.getAugmentedStateSize()))
        YTH = np.zeros((input_dim-2*self.parallel_group_interaction_length, self.getAugmentedStateSize()))
        H = []
        Y = []
        if self.parallel_group_num==0: print("TRAINING: Teacher forcing...")
        for t in range(tl - 1):
            if self.display_output == True and self.parallel_group_num==0:
                print("TRAINING - Teacher forcing: T {:}/{:}, {:2.3f}%".format(t, tl, t/tl*100), end="\r")
            i = np.reshape(train_input_sequence[t+dynamics_length], (-1,1))
            h = np.tanh(W_h @ h + W_in @ i)
            # AUGMENT THE HIDDEN STATE
            h_aug = self.augmentHidden(h)
            # target = np.reshape(train_input_sequence[t + dynamics_length + 1, getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)], (-1,1))
            target = np.reshape(train_input_sequence[t + dynamics_length + 1, getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)], (-1,1))

            H.append(h_aug[:,0])
            Y.append(target[:,0])
            if (t % NORMEVERY == 0):
                H = np.array(H)
                Y = np.array(Y)
                # print(np.shape(H))
                # print(np.shape(Y))
                HTH+=H.T @ H
                YTH+=Y.T @ H
                H = []
                Y = []

        # ADDING THE REMAINNING BATCH
        if (len(H) != 0):
            H = np.array(H)
            Y = np.array(Y)
            # print(np.shape(H))
            # print(np.shape(Y))
            HTH+=H.T @ H
            YTH+=Y.T @ H
            
        if self.parallel_group_num==0: print("\n")

        # COMPUTING THE OUTPUT WEIGHTS
        if self.parallel_group_num==0: print("TRAINING: COMPUTING THE OUTPUT WEIGHTS...")
        if self.parallel_group_num==0: print(np.shape(HTH))
        if self.parallel_group_num==0: print(np.shape(YTH))

        # REGULARISATION
        if self.parallel_group_num==0: print("REGULARISATION...")
        I = np.identity(np.shape(HTH)[1])

        if self.parallel_group_num==0: print("LSTSQ...")
        # pinv_ = scipypinv2(H.T @ H + self.regularization*I)
        # W_out = Y.T @ H @ pinv_
        pinv_ = scipypinv2(HTH + self.regularization*I)
        W_out = YTH @ pinv_
        if self.parallel_group_num==0: print("FINALISING WEIGHTS...")
        self.W_in = W_in
        self.W_h = W_h
        self.W_out = W_out
        if self.parallel_group_num==0: print("COMPUTING PARAMETERS...")
        self.n_trainable_parameters = np.size(self.W_out)
        self.n_model_parameters = np.size(self.W_in) + np.size(self.W_h) + np.size(self.W_out)
        if self.parallel_group_num==0: print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
        if self.parallel_group_num==0: print("Total number of parameters: {}".format(self.n_model_parameters))
        if self.parallel_group_num==0: print("SAVING MODEL...")
        self.saveModel()


    def predictSequence(self, input_sequence):
        W_h = self.W_h
        W_out = self.W_out
        W_in = self.W_in
        dynamics_length = self.dynamics_length
        iterative_prediction_length = self.iterative_prediction_length

        self.reservoir_size, _ = np.shape(W_h)
        N = np.shape(input_sequence)[0]
        
        # PREDICTION LENGTH
        if N != iterative_prediction_length + dynamics_length: raise ValueError("Error! N ({:}) != iterative_prediction_length + dynamics_length {:}, N={:}, iterative_prediction_length={:}, dynamics_length={:}".format(N, iterative_prediction_length+dynamics_length, N, iterative_prediction_length, dynamics_length))

        # H_dyn = np.zeros((dynamics_length, 2*self.reservoir_size, 1))
        h = np.zeros((self.reservoir_size, 1))
        for t in range(dynamics_length):
            if self.display_output == True and self.parallel_group_num == 0:
                print("PREDICTION - Dynamics pre-run: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
            i = np.reshape(input_sequence[t], (-1,1))
            h = np.tanh(W_h @ h + W_in @ i)
            # H_dyn[t] = self.augmentHidden(h)
        if self.parallel_group_num==0: print("\n")

        # target = input_sequence[dynamics_length:, getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)]
        target = input_sequence[dynamics_length:, getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)]

        prediction = []
        for t in range(iterative_prediction_length):
            if self.display_output == True and self.parallel_group_num == 0:
                print("PREDICTION: T {:}/{:}, {:2.3f}%".format(t, iterative_prediction_length, t/iterative_prediction_length*100), end="\r")
            out = W_out @ self.augmentHidden(h)

            prediction.append(out)

            # LOCAL STATE
            global_state = np.zeros((self.RDIM))
            local_state = np.zeros((self.RDIM))
            temp = np.reshape(out.copy(), (-1))
            local_state[self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = temp

            # UPDATING THE GLOBAL STATE - IMPLICIT BARRIER
            comm.Allreduce([local_state, MPI.DOUBLE], [global_state, MPI.DOUBLE], MPI.SUM)
            state_list = Circ(list(global_state.copy()))
            group_start = self.parallel_group_num * self.parallel_group_size
            group_end = group_start + self.parallel_group_size
            pgil = self.parallel_group_interaction_length
            new_input = []
            for i in range(group_start-pgil, group_end+pgil):
                new_input.append(state_list[i].copy())
            new_input = np.array(new_input)

            i = np.reshape(new_input, (-1,1)).copy()
            h = np.tanh(W_h @ h + W_in @ i)
        if self.parallel_group_num==0: print("\n")
        prediction = np.array(prediction)[:,:,0]
        return prediction, target

    def testing(self):
        if self.loadModel()==0:
            self.worker_train_data_path, self.worker_test_data_path = createParallelTrainingData(self)
            self.testingOnTrainingSet()
            self.testingOnTestingSet()
            if rank==0: self.saveResults()

    def testingOnTrainingSet(self):
        num_test_ICS = self.num_test_ICS
        with open(self.worker_test_data_path, "rb") as file:
            data = pickle.load(file)
            testing_ic_indexes = data["testing_ic_indexes"]
            dt = data["dt"]
            del data

        with open(self.worker_train_data_path, "rb") as file:
            data = pickle.load(file)
            train_input_sequence = data["train_input_sequence"][:, :self.input_dim]
            del data
            
        rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred = self.predictIndexes(train_input_sequence, testing_ic_indexes, dt, "TRAIN")
        
        for var_name in getNamesInterestingVars():
            exec("self.{:s}_TRAIN = {:s}".format(var_name, var_name))
        return 0

    def testingOnTestingSet(self):
        num_test_ICS = self.num_test_ICS
        with open(self.worker_test_data_path, "rb") as file:
            data = pickle.load(file)
            testing_ic_indexes = data["testing_ic_indexes"]
            test_input_sequence = data["test_input_sequence"][:, :self.input_dim]
            dt = data["dt"]
            del data
            
        rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred = self.predictIndexes(test_input_sequence, testing_ic_indexes, dt, "TEST")
        
        for var_name in getNamesInterestingVars():
            exec("self.{:s}_TEST = {:s}".format(var_name, var_name))
        return 0

    def predictIndexes(self, input_sequence, ic_indexes, dt, set_name):
        num_test_ICS = self.num_test_ICS
        input_sequence = self.scaler.scaleData(input_sequence, reuse=1)
        local_predictions = []
        local_truths = []
        for ic_num in range(num_test_ICS):
            ic_idx = ic_indexes[ic_num]
            if self.parallel_group_num == 0:
                print("IC {:}/{:}, {:2.3f}%, (ic_idx={:d})".format(ic_num, num_test_ICS, ic_num/num_test_ICS*100, ic_idx))

            input_sequence_ic = input_sequence[ic_idx-self.dynamics_length:ic_idx+self.iterative_prediction_length]
            if self.parallel_group_num == 0: print(np.shape(input_sequence_ic))
            prediction, target = self.predictSequence(input_sequence_ic)
            if self.parallel_group_num == 0: print("SEQUENCE PREDICTED...")
            prediction = self.scaler.descaleDataParallel(prediction, self.parallel_group_interaction_length)
            target = self.scaler.descaleDataParallel(target, self.parallel_group_interaction_length)
            local_predictions.append(prediction)
            local_truths.append(target)


        local_predictions = np.array(local_predictions)
        local_truths = np.array(local_truths)
        comm.Barrier()

        predictions_all_proxy = np.zeros((self.num_test_ICS, self.iterative_prediction_length, self.RDIM))
        truths_all_proxy = np.zeros((self.num_test_ICS, self.iterative_prediction_length, self.RDIM))
        scaler_std_proxy = np.zeros((self.RDIM))

        # SETTING THE LOCAL VALUES
        predictions_all_proxy[:,:,self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = local_predictions
        truths_all_proxy[:,:,self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = local_truths
        scaler_std_proxy[self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = self.scaler.data_std[getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)]


        predictions_all = np.zeros((self.num_test_ICS, self.iterative_prediction_length, self.RDIM)) if(self.parallel_group_num == 0) else None
        truths_all = np.zeros((self.num_test_ICS, self.iterative_prediction_length, self.RDIM)) if(self.parallel_group_num == 0) else None
        scaler_std = np.zeros((self.RDIM)) if(self.parallel_group_num == 0) else None

        comm.Reduce([predictions_all_proxy, MPI.DOUBLE], [predictions_all, MPI.DOUBLE], MPI.SUM, root=0)
        comm.Reduce([truths_all_proxy, MPI.DOUBLE], [truths_all, MPI.DOUBLE], MPI.SUM, root=0)
        comm.Reduce([scaler_std_proxy, MPI.DOUBLE], [scaler_std, MPI.DOUBLE], MPI.SUM, root=0)

        if self.parallel_group_num==0: print("PREDICTION OF ICS FINISHED")
        if(self.parallel_group_num == 0):
            print("MASTER RANK GATHERING PREDICTIONS...")
            # COMPUTING OTHER QUANTITIES
            rmse_all = []
            rmnse_all = []
            num_accurate_pred_005_all = []
            num_accurate_pred_050_all = []
            for ic_num in range(num_test_ICS):
                prediction = predictions_all[ic_num]
                target = truths_all[ic_num]
                rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, abserror = computeErrors(target, prediction, scaler_std)
                rmse_all.append(rmse)
                rmnse_all.append(rmnse)
                num_accurate_pred_005_all.append(num_accurate_pred_005)
                num_accurate_pred_050_all.append(num_accurate_pred_050)
                # PLOTTING ONLY THE FIRST THREE PREDICTIONS
                if ic_num < 3: plotIterativePrediction(self, set_name, target, prediction, rmse, rmnse, ic_idx, dt)

            rmse_all = np.array(rmse_all)
            rmnse_all = np.array(rmnse_all)
            num_accurate_pred_005_all = np.array(num_accurate_pred_005_all)
            num_accurate_pred_050_all = np.array(num_accurate_pred_050_all)

            print("TRAJECTORIES SHAPES:")
            print(np.shape(truths_all))
            print(np.shape(predictions_all))

            rmnse_avg = np.mean(rmnse_all)
            print("AVERAGE RMNSE ERROR: {:}".format(rmnse_avg))

            num_accurate_pred_005_avg = np.mean(num_accurate_pred_005_all)
            print("AVG NUMBER OF ACCURATE 0.05 PREDICTIONS: {:}".format(num_accurate_pred_005_avg))
            num_accurate_pred_050_avg = np.mean(num_accurate_pred_050_all)
            print("AVG NUMBER OF ACCURATE 0.5 PREDICTIONS: {:}".format(num_accurate_pred_050_avg))

            freq_pred, freq_true, sp_true, sp_pred, error_freq = computeFrequencyError(predictions_all, truths_all, dt)

            print("FREQUENCY ERROR: {:}".format(error_freq))

            plotSpectrum(self, sp_true, sp_pred, freq_true, freq_pred, set_name)
        else:
            rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred = None, None, None, None, None, None, None, None, None, None 
        if self.parallel_group_num==0: print("IC INDEXES PREDICTED...")
        return rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred

    def saveResults(self):
        if self.parallel_group_num != 0: raise ValueError("ERROR: ONLY THE MASTER (0) CAN WRITE RESULTS. MEMBER {:d} ATTEMPTED!".format(self.parallel_group_num))

        if self.write_to_log == 1:
            logfile_test = self.saving_path + self.logfile_dir + self.model_name  + "/test.txt"
            writeToTestLogFile(logfile_test, self)

        data = {}
        for var_name in getNamesInterestingVars():
            exec("data['{:s}_TEST'] = self.{:s}_TEST".format(var_name, var_name))
            exec("data['{:s}_TRAIN'] = self.{:s}_TRAIN".format(var_name, var_name))
        data["model_name"] = self.model_name
        data["num_test_ICS"] = self.num_test_ICS
        data_path = self.saving_path + self.results_dir + self.model_name + "/results.pickle"
        with open(data_path, "wb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
        return 0

    def loadModel(self):
        data_path = self.saving_path + self.model_dir + self.model_name + "/data_member_{:d}.pickle".format(self.parallel_group_num)
        try:
            with open(data_path, "rb") as file:
                data = pickle.load(file)
                self.W_out = data["W_out"]
                self.W_in = data["W_in"]
                self.W_h = data["W_h"]
                self.scaler = data["scaler"]
                del data
            return 0
        except:
            print("MODEL {:s} NOT FOUND.".format(data_path))
            return 1

    def saveModel(self):
        if self.parallel_group_num==0: print("Recording time...")
        self.total_training_time = time.time() - self.start_time
        if self.parallel_group_num==0: print("Total training time is {:}".format(self.total_training_time))

        print("MEMORY TRACKING IN MB...")
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss/1024/1024
        self.memory = memory
        print("Script used {:} MB".format(self.memory))

        if self.write_to_log == 1:
            logfile_train = self.saving_path + self.logfile_dir + self.model_name  + "/train.txt"
            writeToTrainLogFile(logfile_train, self)

        data = {
        "memory":self.memory,
        "n_trainable_parameters":self.n_trainable_parameters,
        "n_model_parameters":self.n_model_parameters,
        "total_training_time":self.total_training_time,
        "W_out":self.W_out,
        "W_in":self.W_in,
        "W_h":self.W_h,
        "scaler":self.scaler,
        }
        data_path = self.saving_path + self.model_dir + self.model_name + "/data_member_{:d}.pickle".format(self.parallel_group_num)
        if self.parallel_group_num==0: print("RANK {:d} saving the model... In path {:}".format(self.parallel_group_num, data_path))
        with open(data_path, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
        return 0



