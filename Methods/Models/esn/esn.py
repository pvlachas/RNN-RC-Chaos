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
from functools import partial
print = partial(print, flush=True)

from sklearn.linear_model import Ridge

# MEMORY TRACKING
import psutil

class esn(object):
	def delete(self):
		return 0
		
	def __init__(self, params):
		self.display_output = params["display_output"]

		print("RANDOM SEED: {:}".format(params["worker_id"]))
		np.random.seed(params["worker_id"])

		self.worker_id = params["worker_id"]
		self.input_dim = params["RDIM"]
		self.N_used = params["N_used"]
		self.approx_reservoir_size = params["approx_reservoir_size"]
		self.degree = params["degree"]
		self.radius = params["radius"]
		self.sigma_input = params["sigma_input"]
		self.dynamics_length = params["dynamics_length"]
		self.iterative_prediction_length = params["iterative_prediction_length"]
		self.num_test_ICS = params["num_test_ICS"]
		self.train_data_path = params["train_data_path"]
		self.test_data_path = params["test_data_path"]
		self.fig_dir = params["fig_dir"]
		self.model_dir = params["model_dir"]
		self.logfile_dir = params["logfile_dir"]
		self.write_to_log = params["write_to_log"]
		self.results_dir = params["results_dir"]
		self.saving_path = params["saving_path"]
		self.regularization = params["regularization"]
		self.scaler_tt = params["scaler"]
		self.learning_rate = params["learning_rate"]
		self.number_of_epochs = params["number_of_epochs"]
		self.solver = str(params["solver"])
		##########################################
		self.scaler = scaler(self.scaler_tt)
		self.noise_level = params["noise_level"]
		self.model_name = self.createModelName(params)

		self.reference_train_time = 60*60*(params["reference_train_time"]-params["buffer_train_time"])
		print("Reference train time {:} seconds / {:} minutes / {:} hours.".format(self.reference_train_time, self.reference_train_time/60, self.reference_train_time/60/60))

		os.makedirs(self.saving_path + self.model_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)

	def getKeysInModelName(self):
		keys = {
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
		'worker_id':'WID', 
		}
		return keys


	def createModelName(self, params):
		keys = self.getKeysInModelName()
		str_ = "RNN-esn_" + self.solver
		for key in keys:
			str_ += "-" + keys[key] + "_{:}".format(params[key])
		return str_

	def getSparseWeights(self, sizex, sizey, radius, sparsity, worker_id=1):
		# W = np.zeros((sizex, sizey))
		# Sparse matrix with elements between 0 and 1
		print("WEIGHT INIT")
		W = sparse.random(sizex, sizey, density=sparsity, random_state=worker_id)
		# W = sparse.random(sizex, sizey, density=sparsity, random_state=worker_id, data_rvs=np.random.randn)
		# Sparse matrix with elements between -1 and 1
		# W.data *=2
		# W.data -=1 
		# to print the values do W.A
		print("EIGENVALUE DECOMPOSITION")
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
		dynamics_length = self.dynamics_length
		input_dim = self.input_dim
		N_used = self.N_used

		with open(self.train_data_path, "rb") as file:
			# Pickle the "data" dictionary using the highest protocol available.
			data = pickle.load(file)
			train_input_sequence = data["train_input_sequence"]
			print("Adding noise to the training data. {:} per mille ".format(self.noise_level))
			train_input_sequence = addNoise(train_input_sequence, self.noise_level)
			N_all, dim = np.shape(train_input_sequence)
			if input_dim > dim: raise ValueError("Requested input dimension is wrong.")
			train_input_sequence = train_input_sequence[:N_used, :input_dim]
			dt = data["dt"]
			del data
		print("##Using {:}/{:} dimensions and {:}/{:} samples ##".format(input_dim, dim, N_used, N_all))
		if N_used > N_all: raise ValueError("Not enough samples in the training data.")
		print("SCALING")
		
		train_input_sequence = self.scaler.scaleData(train_input_sequence)

		N, input_dim = np.shape(train_input_sequence)

		# Setting the reservoir size automatically to avoid overfitting
		print("Initializing the reservoir weights...")
		nodes_per_input = int(np.ceil(self.approx_reservoir_size/input_dim))
		self.reservoir_size = int(input_dim*nodes_per_input)
		self.sparsity = self.degree/self.reservoir_size;
		print("NETWORK SPARSITY: {:}".format(self.sparsity))
		print("Computing sparse hidden to hidden weight matrix...")
		W_h = self.getSparseWeights(self.reservoir_size, self.reservoir_size, self.radius, self.sparsity, self.worker_id)

		# Initializing the input weights
		print("Initializing the input weights...")
		W_in = np.zeros((self.reservoir_size, input_dim))
		q = int(self.reservoir_size/input_dim)
		for i in range(0, input_dim):
			W_in[i*q:(i+1)*q,i] = self.sigma_input * (-1 + 2*np.random.rand(q))

		# TRAINING LENGTH
		tl = N - dynamics_length

		print("TRAINING: Dynamics prerun...")
		# H_dyn = np.zeros((dynamics_length, self.getAugmentedStateSize(), 1))
		h = np.zeros((self.reservoir_size, 1))
		for t in range(dynamics_length):
			if self.display_output == True:
				print("TRAINING - Dynamics prerun: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
			i = np.reshape(train_input_sequence[t], (-1,1))
			h = np.tanh(W_h @ h + W_in @ i)
			# H_dyn[t] = self.augmentHidden(h)

		print("\n")

		# # FOR PLOTTING THE DYNAMICS
		# dyn_plot_max = 1000
		# H_dyn_plot = H_dyn[:,:dyn_plot_max,0]
		# fig_path = self.saving_path + self.fig_dir + self.model_name + "/H_dyn_prerun_plot.png"
		# plt.plot(H_dyn_plot)
		# plt.title('Dynamics prerun')
		# plt.savefig(fig_path)
		# plt.close()

		if self.solver == "pinv":
			NORMEVERY = 10
			HTH = np.zeros((self.getAugmentedStateSize(), self.getAugmentedStateSize()))
			YTH = np.zeros((input_dim, self.getAugmentedStateSize()))
		H = []
		Y = []

		print("TRAINING: Teacher forcing...")
		
		for t in range(tl - 1):
			if self.display_output == True:
				print("TRAINING - Teacher forcing: T {:}/{:}, {:2.3f}%".format(t, tl, t/tl*100), end="\r")
			i = np.reshape(train_input_sequence[t+dynamics_length], (-1,1))
			h = np.tanh(W_h @ h + W_in @ i)
			# AUGMENT THE HIDDEN STATE
			h_aug = self.augmentHidden(h)
			H.append(h_aug[:,0])
			target = np.reshape(train_input_sequence[t+dynamics_length+1], (-1,1))
			Y.append(target[:,0])
			if self.solver == "pinv" and (t % NORMEVERY == 0):
				# Batched approach used in the pinv case
				H = np.array(H)
				Y = np.array(Y)
				HTH += H.T @ H
				YTH += Y.T @ H
				H = []
				Y = []

		if self.solver == "pinv" and (len(H) != 0):
			# ADDING THE REMAINING BATCH
			H = np.array(H)
			Y = np.array(Y)
			HTH+=H.T @ H
			YTH+=Y.T @ H
			print("TEACHER FORCING ENDED.")
			print(np.shape(H))
			print(np.shape(Y))
			print(np.shape(HTH))
			print(np.shape(YTH))
		else:
			print("TEACHER FORCING ENDED.")
			print(np.shape(H))
			print(np.shape(Y))

		
		print("\nSOLVER used to find W_out: {:}. \n\n".format(self.solver))

		if self.solver == "pinv":
			"""
			Learns mapping H -> Y with Penrose Pseudo-Inverse
			"""
			I = np.identity(np.shape(HTH)[1])	
			pinv_ = scipypinv2(HTH + self.regularization*I)
			W_out = YTH @ pinv_

		elif self.solver in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]:
			"""
			Learns mapping H -> Y with Ridge Regression
			"""
			ridge = Ridge(alpha=self.regularization, fit_intercept=False, normalize=False, copy_X=True, solver=self.solver)
			# print(np.shape(H))
			# print(np.shape(Y))
			# print("##")
			ridge.fit(H, Y) 
			W_out = ridge.coef_

		else:
			raise ValueError("Undefined solver.")

		print("FINALISING WEIGHTS...")
		self.W_in = W_in
		self.W_h = W_h
		self.W_out = W_out
		
		print("COMPUTING PARAMETERS...")
		self.n_trainable_parameters = np.size(self.W_out)
		self.n_model_parameters = np.size(self.W_in) + np.size(self.W_h) + np.size(self.W_out)
		print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
		print("Total number of parameters: {}".format(self.n_model_parameters))
		print("SAVING MODEL...")
		self.saveModel()

	def isWallTimeLimit(self):
		training_time = time.time() - self.start_time
		if training_time > self.reference_train_time:
			print("## Maximum train time reached. ##")
			return True
		else:
			return False

	def predictSequence(self, input_sequence):
		W_h = self.W_h
		W_out = self.W_out
		W_in = self.W_in
		dynamics_length = self.dynamics_length
		iterative_prediction_length = self.iterative_prediction_length

		self.reservoir_size, _ = np.shape(W_h)
		N = np.shape(input_sequence)[0]
		
		# PREDICTION LENGTH
		if N != iterative_prediction_length + dynamics_length: raise ValueError("Error! N != iterative_prediction_length + dynamics_length")


		prediction_warm_up = []
		h = np.zeros((self.reservoir_size, 1))
		for t in range(dynamics_length):
			if self.display_output == True:
				print("PREDICTION - Dynamics pre-run: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
			i = np.reshape(input_sequence[t], (-1,1))
			h = np.tanh(W_h @ h + W_in @ i)
			out = W_out @ self.augmentHidden(h)
			prediction_warm_up.append(out)

		print("\n")

		target = input_sequence[dynamics_length:]
		prediction = []
		for t in range(iterative_prediction_length):
			if self.display_output == True:
				print("PREDICTION: T {:}/{:}, {:2.3f}%".format(t, iterative_prediction_length, t/iterative_prediction_length*100), end="\r")
			out = W_out @ self.augmentHidden(h)
			prediction.append(out)
			i = out
			h = np.tanh(W_h @ h + W_in @ i)
		print("\n")

		prediction = np.array(prediction)[:,:,0]
		prediction_warm_up = np.array(prediction_warm_up)[:,:,0]

		target_augment = input_sequence
		prediction_augment = np.concatenate((prediction_warm_up, prediction), axis=0)

		return prediction, target, prediction_augment, target_augment

	def predictSequenceMemoryCapacity(self, input_sequence, target_sequence):
		W_h = self.W_h
		W_out = self.W_out
		W_in = self.W_in
		dynamics_length = self.dynamics_length
		iterative_prediction_length = self.iterative_prediction_length

		self.reservoir_size, _ = np.shape(W_h)
		N = np.shape(input_sequence)[0]
		
		# PREDICTION LENGTH
		if N != iterative_prediction_length + dynamics_length: raise ValueError("Error! N != iterative_prediction_length + dynamics_length")

		h = np.zeros((self.reservoir_size, 1))
		for t in range(dynamics_length):
			if self.display_output == True:
				print("PREDICTION - Dynamics pre-run: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
			i = np.reshape(input_sequence[t], (-1,1))
			h = np.tanh(W_h @ h + W_in @ i)
		print("\n")

		target = target_sequence
		prediction = []
		signal = []
		for t in range(dynamics_length, dynamics_length+iterative_prediction_length):
			if self.display_output == True:
				print("PREDICTION: T {:}/{:}, {:2.3f}%".format(t, iterative_prediction_length, t/iterative_prediction_length*100), end="\r")
			signal.append(i)
			out = W_out @ self.augmentHidden(h)
			prediction.append(out)
			# TEACHER FORCING
			i = np.reshape(input_sequence[t], (-1,1))
			# i = out
			h = np.tanh(W_h @ h + W_in @ i)
		print("\n")
		prediction = np.array(prediction)[:,:,0]
		target = np.array(target)
		signal = np.array(signal)
		return prediction, target, signal

	def testing(self):
		if self.loadModel()==0:
			self.testingOnTrainingSet()
			self.testingOnTestingSet()
			self.saveResults()
		return 0

	def testingOnTrainingSet(self):
		num_test_ICS = self.num_test_ICS
		with open(self.test_data_path, "rb") as file:
			data = pickle.load(file)
			testing_ic_indexes = data["testing_ic_indexes"]
			dt = data["dt"]
			del data

		with open(self.train_data_path, "rb") as file:
			data = pickle.load(file)
			train_input_sequence = data["train_input_sequence"][:, :self.input_dim]
			del data
			
		rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred = self.predictIndexes(train_input_sequence, testing_ic_indexes, dt, "TRAIN")
		
		for var_name in getNamesInterestingVars():
			exec("self.{:s}_TRAIN = {:s}".format(var_name, var_name))
		return 0

	def testingOnTestingSet(self):
		num_test_ICS = self.num_test_ICS
		with open(self.test_data_path, "rb") as file:
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
		predictions_all = []
		truths_all = []
		rmse_all = []
		rmnse_all = []
		num_accurate_pred_005_all = []
		num_accurate_pred_050_all = []
		for ic_num in range(num_test_ICS):
			if self.display_output == True:
				print("IC {:}/{:}, {:2.3f}%".format(ic_num, num_test_ICS, ic_num/num_test_ICS*100))
			ic_idx = ic_indexes[ic_num]
			input_sequence_ic = input_sequence[ic_idx-self.dynamics_length:ic_idx+self.iterative_prediction_length]
			prediction, target, prediction_augment, target_augment = self.predictSequence(input_sequence_ic)
			prediction = self.scaler.descaleData(prediction)
			target = self.scaler.descaleData(target)
			rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, abserror = computeErrors(target, prediction, self.scaler.data_std)
			predictions_all.append(prediction)
			truths_all.append(target)
			rmse_all.append(rmse)
			rmnse_all.append(rmnse)
			num_accurate_pred_005_all.append(num_accurate_pred_005)
			num_accurate_pred_050_all.append(num_accurate_pred_050)
			# PLOTTING ONLY THE FIRST THREE PREDICTIONS
			if ic_num < 3: plotIterativePrediction(self, set_name, target, prediction, rmse, rmnse, ic_idx, dt, target_augment, prediction_augment, self.dynamics_length)

		predictions_all = np.array(predictions_all)
		truths_all = np.array(truths_all)
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
		return rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred

	def saveResults(self):

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
		data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
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
		print("Recording time...")
		self.total_training_time = time.time() - self.start_time
		print("Total training time is {:}".format(self.total_training_time))

		print("MEMORY TRACKING IN MB...")
		process = psutil.Process(os.getpid())
		memory = process.memory_info().rss/1024/1024
		self.memory = memory
		print("Script used {:} MB".format(self.memory))
		print("SAVING MODEL...")

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
		data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
		with open(data_path, "wb") as file:
			pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
			del data
		return 0

