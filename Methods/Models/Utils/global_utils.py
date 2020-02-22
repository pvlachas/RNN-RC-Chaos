#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import pickle
import io
import os


def getNamesInterestingVars():
	# THE MODEL SHOULD OUTPUT THE FOLLOWING VARIABLES:
	var_names = [
		'rmnse_avg',
		'num_accurate_pred_005_avg',
		'num_accurate_pred_050_avg',
		'error_freq',
		'predictions_all',
		'truths_all',
		'freq_pred',
		'freq_true',
		'sp_true',
		'sp_pred',
	]
	return var_names

def writeToTrainLogFile(logfile_train, model):
	with io.open(logfile_train, 'a+') as f:
		f.write("model_name:" + str(model.model_name)
			+ ":memory:" +"{:.2f}".format(model.memory)
			+ ":total_training_time:" + "{:.2f}".format(model.total_training_time) \
			+ ":n_model_parameters:" + str(model.n_model_parameters) \
			+ ":n_trainable_parameters:" + str(model.n_trainable_parameters) \
			+ "\n"
			)
	return 0

def writeToTestLogFile(logfile_test, model):
	with io.open(logfile_test, 'a+') as f:
		f.write("model_name:" + str(model.model_name)
			+ ":num_test_ICS:" + "{:.2f}".format(model.num_test_ICS)
			+ ":num_accurate_pred_005_avg_TEST:" + "{:.2f}".format(model.num_accurate_pred_005_avg_TEST)
			+ ":num_accurate_pred_050_avg_TEST:" + "{:.2f}".format(model.num_accurate_pred_050_avg_TEST) \
			+ ":num_accurate_pred_005_avg_TRAIN:" + "{:.2f}".format(model.num_accurate_pred_005_avg_TRAIN)
			+ ":num_accurate_pred_050_avg_TRAIN:" + "{:.2f}".format(model.num_accurate_pred_050_avg_TRAIN) \
			+ ":error_freq_TRAIN:" + "{:.2f}".format(model.error_freq_TRAIN) \
			+ ":error_freq_TEST:" + "{:.2f}".format(model.error_freq_TEST) \
			+ "\n"
			)
	return 0

def getReferenceTrainingTime(rtt, btt):
    reference_train_time = 60*60*(rtt-btt)
    print("Reference train time {:} seconds / {:} minutes / {:} hours.".format(rtt, rtt/60, rtt/60/60))
    return reference_train_time


def countTrainableParams(layers):
    temp = 0
    for layer in layers:
        temp+= sum(p.numel() for p in layer.parameters() if p.requires_grad)
    return temp

def printTime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    h = int(h)
    m = int(m)
    s = int(s)
    print("Time passed: {:d}:{:02d}:{:02d}".format(h, m, s))
    return 0

def countParams(layers):
    temp = 0
    for layer in layers:
        temp+= sum(p.numel() for p in layer.parameters())
    return temp

def splitTrajectories(trajectories, params):
    N_train = int(np.floor(trajectories.size(0) * params["train_valid_ratio"]))
    trajectories_train = trajectories[:N_train]
    trajectories_val = trajectories[N_train:]
    return trajectories_train, trajectories_val


def replaceNaN(data):
	data[np.isnan(data)]=float('Inf')
	return data

def computeErrors(target, prediction, std):
	prediction = replaceNaN(prediction)
	# SQUARE ERROR
	abserror = np.mean(np.abs(target-prediction), axis=1)
	# SQUARE ERROR
	serror = np.square(target-prediction)
	# MEAN (over-space) SQUARE ERROR
	mse = np.mean(serror, axis=1)
	# ROOT MEAN SQUARE ERROR
	rmse = np.sqrt(mse)
	# NORMALIZED SQUARE ERROR
	nserror = serror/np.square(std)
	# MEAN (over-space) NORMALIZED SQUARE ERROR
	mnse = np.mean(nserror, axis=1)
	# ROOT MEAN NORMALIZED SQUARE ERROR
	rmnse = np.sqrt(mnse)
	num_accurate_pred_005 = getNumberOfAccuratePredictions(rmnse, 0.05)
	num_accurate_pred_050 = getNumberOfAccuratePredictions(rmnse, 0.5)
	return rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, abserror

def computeFrequencyError(predictions_all, truths_all, dt):
	sp_pred, freq_pred = computeSpectrum(predictions_all, dt)
	sp_true, freq_true = computeSpectrum(truths_all, dt)
	error_freq = np.mean(np.abs(sp_pred - sp_true))
	return freq_pred, freq_true, sp_true, sp_pred, error_freq

def addNoise(data, percent):
	std_data = np.std(data, axis=0)
	std_data = np.reshape(std_data, (1, -1))
	std_data = np.repeat(std_data, np.shape(data)[0], axis=0)
	noise = np.multiply(np.random.randn(*np.shape(data)), percent/1000.0*std_data)
	data += noise
	return data

class scaler(object):
	def __init__(self, tt):
		self.tt = tt
		self.data_min = 0
		self.data_max = 0
		self.data_mean = 0
		self.data_std = 0       

	def scaleData(self, input_sequence, reuse=None):
		# data_mean = np.mean(train_input_sequence,0)
		# data_std = np.std(train_input_sequence,0)
		# train_input_sequence = (train_input_sequence-data_mean)/data_std
		if reuse == None:
			self.data_mean = np.mean(input_sequence,0)
			self.data_std = np.std(input_sequence,0)
			self.data_min = np.min(input_sequence,0)
			self.data_max = np.max(input_sequence,0)
		if self.tt == "MinMaxZeroOne":
			input_sequence = np.array((input_sequence-self.data_min)/(self.data_max-self.data_min))
		elif self.tt == "Standard" or self.tt == "standard":
			input_sequence = np.array((input_sequence-self.data_mean)/self.data_std)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence

	def descaleData(self, input_sequence):
		if self.tt == "MinMaxZeroOne":
			input_sequence = np.array(input_sequence*(self.data_max - self.data_min) + self.data_min)
		elif self.tt == "Standard" or self.tt == "standard":
			input_sequence = np.array(input_sequence*self.data_std.T + self.data_mean)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence

	def descaleDataParallel(self, input_sequence, interaction_length):
		# Descaling in the parallel model requires to substract the neighboring points from the scaler
		if self.tt == "MinMaxZeroOne":
			input_sequence = np.array(input_sequence*(self.data_max[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)] - self.data_min[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)]) + self.data_min[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)])
		elif self.tt == "Standard" or self.tt == "standard":
			input_sequence = np.array(input_sequence*self.data_std[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)].T + self.data_mean[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)])
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence



def computeSpectrum(data_all, dt):
	# Of the form [n_ics, T, n_dim]
	spectrum_db = []
	for data in data_all:
		data = np.transpose(data)
		for d in data:
			freq, s_dbfs = dbfft(d, 1/dt)
			spectrum_db.append(s_dbfs)
	spectrum_db = np.array(spectrum_db).mean(axis=0)
	return spectrum_db, freq


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

def getNumberOfAccuratePredictions(nerror, tresh=0.05):
	nerror_bool = nerror < tresh
	n_max = np.shape(nerror)[0]
	n = 0
	while nerror_bool[n] == True:
		n += 1
		if n == n_max: break
	return n


def addWhiteNoise(data, noise_level):
	std_ = np.std(data, axis=0)
	std_ = np.array(std_).flatten(-1)
	data += np.random.randn(*data.shape)*std_*noise_level/1000.0
	return data


def computeNumberOfModelParameters(variables):
	total_parameters = 0
	for variable in variables:
		shape = variable.get_shape()
		variable_parametes = 1
		for dim in shape:
			variable_parametes *= dim.value
		total_parameters += variable_parametes
	return total_parameters

def isZeroOrNone(var):
	return (var==0 or var==None or var == False or var == str(0))

def stackSequenceData(sequence_data, sequence_length, prediction_length, subsample_seq):
	stacked_input_data = []
	stacked_target_data = []
	if(subsample_seq!=1): print("SEQUENTIALL SUBSAMPLING, ONLY USE IT IN STATE-LESS RNNs WITH LARGE DATA-SETS")
	n = getFirstDataDimension(sequence_data)
	for i in range(0, n - sequence_length - prediction_length, subsample_seq):
		sequence = sequence_data[i:(i+sequence_length), :]
		prediction = sequence_data[(i+sequence_length):(i+sequence_length+prediction_length), :]
		stacked_input_data.append(sequence)
		stacked_target_data.append(prediction)
	return stacked_input_data, stacked_target_data


def stackParallelSequenceData(sequence_data, sequence_length, prediction_length, subsample_seq, parallel_group_interaction_length):
	stacked_input_data = []
	stacked_target_data = []
	pgil = parallel_group_interaction_length
	if(subsample_seq!=1): print("SEQUENTIALL SUBSAMPLING, ONLY USE IT IN STATE-LESS RNNs WITH LARGE DATA-SETS")
	n = getFirstDataDimension(sequence_data)
	for i in range(0, n - sequence_length - prediction_length, subsample_seq):
		sequence = sequence_data[i:(i+sequence_length), :]
		prediction = sequence_data[(i+sequence_length):(i+sequence_length+prediction_length), getFirstActiveIndex(pgil):getLastActiveIndex(pgil)]
		stacked_input_data.append(sequence)
		stacked_target_data.append(prediction)
	return stacked_input_data, stacked_target_data


def getFirstActiveIndex(parallel_group_interaction_length):
	if parallel_group_interaction_length > 0:
		return parallel_group_interaction_length
	else:
		return 0

def getLastActiveIndex(parallel_group_interaction_length):
	if parallel_group_interaction_length > 0:
		return -parallel_group_interaction_length
	else:
		return None

def getFirstDataDimension(var):
	if isinstance(var, (list,)):
		dim = len(var)
	elif type(var) == np.ndarray:
		dim = np.shape(var)[0]
	elif  type(var) == np.matrix:
		raise ValueError("Variable is a matrix. NOT ALLOWED!")
	else:
		raise ValueError("Variable not a list or a numpy array. No dimension to compute!")
	return dim

def divideData(data, train_val_ratio):
	n_samples = getFirstDataDimension(data)
	n_train = int(n_samples*train_val_ratio)
	data_train = data[:n_train]
	data_val = data[n_train:]
	return data_train, data_val

def createTrainingDataBatches(input_train, target_train, batch_size):
	n_samples = getFirstDataDimension(input_train)
	input_train_batches = []
	target_train_batches = []
	n_batches = int(n_samples/batch_size)
	for i in range(n_batches):
		input_train_batches.append(input_train[batch_size*i:batch_size*i+batch_size])
		target_train_batches.append(target_train[batch_size*i:batch_size*i+batch_size])
	return input_train_batches, target_train_batches, n_batches

def subsample(data, max_samples):
	# Subsampling the sequence to ensure that the computations do not explode
	n_samples = getFirstDataDimension(data)
	if n_samples>max_samples:
		step = int(np.floor(n_samples/max_samples))
		if step == 1:
			data = data[:max_samples]
		else:
			data = data[::step][:max_samples]
	return data


def getESNParser(parser):
	parser.add_argument("--mode", help="train, test, all", type=str, required=True)
	parser.add_argument("--system_name", help="system_name", type=str, required=True)
	parser.add_argument("--write_to_log", help="write_to_log", type=int, required=True)
	parser.add_argument("--N", help="N", type=int, required=True)
	parser.add_argument("--N_used", help="N_used", type=int, required=True)
	parser.add_argument("--RDIM", help="RDIM", type=int, required=True)
	parser.add_argument("--approx_reservoir_size", help="approx_reservoir_size", type=int, required=True)
	parser.add_argument("--degree", help="degree", type=float, required=True)
	parser.add_argument("--radius", help="radius", type=float, required=True)
	parser.add_argument("--sigma_input", help="sigma_input", type=float, required=True)
	parser.add_argument("--regularization", help="regularization", type=float, required=True)
	parser.add_argument("--dynamics_length", help="dynamics_length", type=int, required=True)
	parser.add_argument("--iterative_prediction_length", help="iterative_prediction_length", type=int, required=True)
	parser.add_argument("--num_test_ICS", help="num_test_ICS", type=int, required=True)
	parser.add_argument("--scaler", help="scaler", type=str, required=True)
	parser.add_argument("--noise_level", help="noise level per mille in the training data", type=int, default=0, required=True)
	parser.add_argument("--display_output", help="control the verbosity level of output , default True", type=int, required=False, default=1)
	parser.add_argument("--mem_cap", help="candidate for memory capacity calculation", type=int, required=False, default=0)
	parser.add_argument("--mem_cap_delay", help="mem_cap_delay", type=int, required=False, default=0)
	parser.add_argument("--learning_rate", help="learning rate for gradient descent", type=float, required=False, default=1e-6)
	parser.add_argument("--number_of_epochs", help="number of epochs", type=int, required=False, default=10000)
	parser.add_argument("--solver", help="solver used to learn mapping H -> Y, it can be [pinv, saga, gd]", type=str, required=False, default="pinv")
	parser.add_argument("--reference_train_time", help="The reference train time in hours", type=float, default=24)
	parser.add_argument("--buffer_train_time", help="The buffer train time to save the model in hours", type=float, default=0.5)

	return parser

def getMLPParser(parser):
	parser.add_argument("--mode", help="train, test, all", type=str, required=True)
	parser.add_argument("--system_name", help="system_name", type=str, required=True)
	parser.add_argument("--write_to_log", help="write_to_log", type=int, required=True)
	parser.add_argument("--N", help="N", type=int, required=True)
	parser.add_argument("--N_used", help="N_used", type=int, required=True)
	parser.add_argument("--RDIM", help="RDIM", type=int, required=True)
	parser.add_argument("--initializer", help="initializer", type=str, required=True)
	parser.add_argument("--mlp_num_layers", help="mlp_num_layers", type=int, required=True)
	parser.add_argument("--mlp_size_layers", help="mlp_size_layers", type=int, required=True)
	parser.add_argument("--mlp_activation_str", help="mlp_activation_str", type=str, required=True)
	parser.add_argument("--prediction_length", help="prediction_length", type=int, required=True)
	parser.add_argument("--sequence_length", help="sequence_length", type=int, required=True)
	parser.add_argument("--scaler", help="scaler", type=str, required=True)
	parser.add_argument("--noise_level", help="noise level per mille in the training data", type=int, default=0, required=True)
	parser.add_argument("--learning_rate", help="learning_rate", type=float, required=True)
	parser.add_argument("--batch_size", help="batch_size", type=int, required=True)
	parser.add_argument("--batched_valtrain", help="batched_valtrain", type=int, required=True)
	parser.add_argument("--overfitting_patience", help="overfitting_patience", type=int, required=True)
	parser.add_argument("--training_min_epochs", help="training_min_epochs", type=int, required=True)
	parser.add_argument("--max_epochs", help="max_epochs", type=int, required=True)
	parser.add_argument("--num_rounds", help="num_rounds", type=int, required=True)
	parser.add_argument("--regularization", help="regularization", type=float, required=True)
	parser.add_argument("--keep_prob", help="keep_prob", type=float, required=True)
	parser.add_argument("--train_val_ratio", help="train_val_ratio", type=float, required=True)
	parser.add_argument("--retrain", help="retrain", type=int, required=True)
	parser.add_argument("--subsample", help="subsample", type=int, required=True)
	parser.add_argument("--num_test_ICS", help="num_test_ICS", type=int, required=True)
	parser.add_argument("--iterative_prediction_length", help="iterative_prediction_length", type=int, required=True)
	parser.add_argument("--display_output", help="control the verbosity level of output , default True", type=int, required=False, default=1)
	return parser



def getRNNStatefullParser(parser):
	parser.add_argument("--mode", help="train, test, all", type=str, required=True)
	parser.add_argument("--system_name", help="system_name", type=str, required=True)
	parser.add_argument("--write_to_log", help="write_to_log", type=int, required=True)
	parser.add_argument("--N", help="N", type=int, required=True)
	parser.add_argument("--N_used", help="N_used", type=int, required=True)
	parser.add_argument("--RDIM", help="RDIM", type=int, required=True)
	parser.add_argument("--initializer", help="initializer", type=str, required=True)
	parser.add_argument("--train_val_ratio", help="train_val_ratio", type=float, required=True)
	parser.add_argument("--rnn_num_layers", help="rnn_num_layers", type=int, required=True)
	parser.add_argument("--rnn_size_layers", help="rnn_size_layers", type=int, required=True)
	parser.add_argument("--rnn_activation_str", help="rnn_activation_str", type=str, required=False)
	parser.add_argument("--dropout_keep_prob", help="dropout_keep_prob", type=float, required=True)
	parser.add_argument("--zoneout_keep_prob", help="zoneout_keep_prob", type=float, required=True)
	parser.add_argument("--prediction_length", help="prediction_length", type=int, required=True)
	parser.add_argument("--sequence_length", help="sequence_length", type=int, required=True)
	parser.add_argument("--hidden_state_propagation_length", help="hidden_state_propagation_length", type=int, required=True)
	parser.add_argument("--scaler", help="scaler", type=str, required=True)
	parser.add_argument("--noise_level", help="noise level per mille in the training data", type=int, default=0, required=True)

	parser.add_argument("--learning_rate", help="learning_rate", type=float, required=True)
	parser.add_argument("--batch_size", help="batch_size", type=int, required=True)
	parser.add_argument("--overfitting_patience", help="overfitting_patience", type=int, required=True)
	parser.add_argument("--training_min_epochs", help="training_min_epochs", type=int, required=True)
	parser.add_argument("--max_epochs", help="max_epochs", type=int, required=True)
	parser.add_argument("--num_rounds", help="num_rounds", type=int, required=True)
	parser.add_argument("--regularization", help="regularization", type=float, required=True)
	parser.add_argument("--retrain", help="retrain", type=int, required=True)
	parser.add_argument("--subsample", help="subsample", type=int, required=True)
	parser.add_argument("--num_test_ICS", help="num_test_ICS", type=int, required=True)
	parser.add_argument("--iterative_prediction_length", help="iterative_prediction_length", type=int, required=True)
	parser.add_argument("--rnn_cell_type", help="type of the rnn cell", type=str, required=True)
	parser.add_argument("--unitary_capacity", help="unitary_capacity", type=int, required=False)
	parser.add_argument("--unitary_cplex", help="unitary_cplex", type=int, required=False)
	parser.add_argument("--display_output", help="control the verbosity level of output , default True", type=int, required=False, default=1)
	parser.add_argument("--mem_cap", help="candidate for memory capacity calculation", type=int, required=False, default=0)
	parser.add_argument("--mem_cap_delay", help="mem_cap_delay", type=int, required=False, default=0)
	parser.add_argument("--trackHiddenState", help="trackHiddenState", type=int, required=False)
	parser.add_argument("--reference_train_time", help="The reference train time in hours", type=float, default=24)
	parser.add_argument("--buffer_train_time", help="The buffer train time to save the model in hours", type=float, default=0.5)
	return parser

def getMLPParallelParser(parser):
	parser = getMLPParser(parser)
	parser.add_argument("--num_parallel_groups", help="groups in the output for parallelization in the spatiotemporal domain. must be divisor of the input dimension", type=int, required=True)
	parser.add_argument("--parallel_group_interaction_length", help="The interaction length of each group. 0-rdim/2", type=int, required=True)
	return parser

def getESNParallelParser(parser):
	parser = getESNParser(parser)
	parser.add_argument("--num_parallel_groups", help="groups in the output for parallelization in the spatiotemporal domain. must be divisor of the input dimension", type=int, required=True)
	parser.add_argument("--parallel_group_interaction_length", help="The interaction length of each group. 0-rdim/2", type=int, required=True)
	return parser


def getRNNStatefullParallelParser(parser):
	parser = getRNNStatefullParser(parser)
	parser.add_argument("--num_parallel_groups", help="groups in the output for parallelization in the spatiotemporal domain. must be divisor of the input dimension", type=int, required=True)
	parser.add_argument("--parallel_group_interaction_length", help="The interaction length of each group. 0-rdim/2", type=int, required=True)
	return parser


# UTILITIES FOR PARALLEL MODELS
def createParallelTrainingData(model):
	# GROUP NUMMER = WORKER ID
	group_num = model.parallel_group_num
	group_start = group_num * model.parallel_group_size
	group_end = group_start + model.parallel_group_size
	pgil = model.parallel_group_interaction_length
	training_path_group = reformatParallelGroupDataPath(model, model.main_train_data_path, group_num, pgil)
	testing_path_group = reformatParallelGroupDataPath(model, model.main_test_data_path, group_num, pgil)
	if(not os.path.isfile(training_path_group)):
		print("## Generating data for group {:d}-{:d} ##".format(group_start, group_end))
		with open(model.main_train_data_path, "rb") as file:
			data = pickle.load(file)
			train_sequence = data["train_input_sequence"][:, :model.RDIM]
			dt = data["dt"]
			del data
		train_sequence_group = createParallelGroupTrainingSequence(group_num, group_start, group_end, pgil, train_sequence)
		data = {"train_input_sequence":train_sequence_group, "dt":dt}
		with open(training_path_group, "wb") as file:
			# Pickle the "data" dictionary using the highest protocol available.
			pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
			del data
	else:
		print("Training data file already exist.")

	if(not os.path.isfile(testing_path_group)):
		print("## Generating data for group {:d}-{:d} ##".format(group_start, group_end))
		with open(model.main_test_data_path, "rb") as file:
			data = pickle.load(file)
			test_sequence = data["test_input_sequence"][:, :model.RDIM]
			testing_ic_indexes = data["testing_ic_indexes"]
			del data
		test_sequence_group = createParallelGroupTrainingSequence(group_num, group_start, group_end, pgil, test_sequence)
		data = {"test_input_sequence":test_sequence_group, "testing_ic_indexes":testing_ic_indexes, "dt":dt}
		with open(testing_path_group, "wb") as file:
			# Pickle the "data" dictionary using the highest protocol available.
			pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
			del data
	else:
		print("Testing data file already exist.")

	return training_path_group, testing_path_group

def createParallelGroupTrainingSequence(gn, gs, ge, ll, sequence):
	sequence = np.transpose(sequence)
	sequence_group = []
	sequence = Circ(sequence)
	print("Indexes considered {:d}-{:d}".format(gs-ll, ge+ll))
	for i in range(gs-ll, ge+ll):
		sequence_group.append(sequence[i])
	sequence_group = np.transpose(sequence_group)
	return sequence_group

def reformatParallelGroupDataPath(model, path, gn, ll):
	# Last 7 string objects are .pickle
	last = 7
	path_ = path[:-last] + "_G{:d}-from-{:d}_GS{:d}_GIL{:d}".format(gn, model.num_parallel_groups, model.parallel_group_size, ll) + path[-last:]
	return path_

class Circ(list):
	def __getitem__(self, idx):
		return super(Circ, self).__getitem__(idx % len(self))

