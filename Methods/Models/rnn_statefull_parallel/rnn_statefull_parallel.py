#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import sys
import scipy.linalg
import cmath as cmath
from eunn_cell import *
from zoneout_wrapper import *

from utils import *
from plotting_utils import *
from global_utils import *
import pickle
import tensorflow as tf

# MEMORY TRACKING
import psutil

import os
import random
import time
import signal

from functools import partial
print = partial(print, flush=True)

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

tf_activations = {"relu": tf.nn.relu, "tanh": tf.tanh, "sigmoid": tf.sigmoid, "identity": tf.identity, "softplus":tf.nn.softplus, "softmax":tf.nn.softmax}
tf_initializers = {"xavier": tf.contrib.layers.xavier_initializer(), "normal": tf.truncated_normal_initializer()}

TFDTYPE = tf.float64
TFCDTYPE = tf.complex128 if TFDTYPE==tf.float64 else tf.complex64

NPDTYPE = np.float64 if TFDTYPE==tf.float64 else np.float32
NPCDTYPE = np.complex128 if TFCDTYPE==tf.complex128 else np.complex64

# print("DATA TYPES: ")
# print(TFDTYPE)
# print(TFCDTYPE)
# print(NPDTYPE)
# print(NPCDTYPE)

# FOR STATEFULL RNN - IMPLEMENTATION OF TRUNCATED BACKPROPAGATION(sl, sl) - jump sl steps, backpropagate sl steps!
# LIMITATION: MAXIMUM BATCH_SIZE IS THE SEQUENCE LENGTH

class rnn_statefull_parallel(object):
    
    def delete(self):
        if self.parallel_group_num==0: print("Resetting default graph!")
        tf.reset_default_graph()

    def rnn_layer(self, cell_type, name, input, input_size, num_hidden, activation_str, ln, initial_hidden_state):
        scope_name = 'rnn_layer_' + name + '_' + str(ln)
        with tf.variable_scope(scope_name, initializer = tf_initializers[self.initializer]):
            # 'input' is a tensor of shape [batch_size, max_time, input_dim]
            if cell_type == 'lstm':
                cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=num_hidden, forget_bias=1.0, state_is_tuple = True, activation=tf_activations[activation_str])
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._dropout_keep_prob)
                cell = ZoneoutWrapper(cell, zoneout_drop_prob=1-self._zoneout_keep_prob, is_training=self._is_training, dtype=self.dtype_tf)
            elif cell_type == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(name='gru_cell', num_units=num_hidden, activation=tf_activations[activation_str])
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._dropout_keep_prob)
                cell = ZoneoutWrapper(cell, zoneout_drop_prob=1-self._zoneout_keep_prob, is_training=self._is_training, dtype=self.dtype_tf)
            elif cell_type == 'plain':
                cell = tf.contrib.rnn.BasicRNNCell(num_units=num_hidden, activation=tf_activations[activation_str])
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._dropout_keep_prob)
                cell = ZoneoutWrapper(cell, zoneout_drop_prob=1-self._zoneout_keep_prob, is_training=self._is_training, dtype=self.dtype_tf)
            elif cell_type == 'unitary':
                cell = EUNNCell(num_units=num_hidden,cplex=self.unitary_cplex,capacity=self.unitary_capacity, dtype=self.dtype_tf, ln=ln)
            else:
                raise ValueError("Invalid cell type provided!")
            outputs, last_state = tf.nn.dynamic_rnn(cell, input, dtype=self.cdtype_tf, initial_state=initial_hidden_state)
            outputs = tf.real(outputs)
        return outputs, last_state

    def mlp_unfolded_layer(self, name, input, input_size, layer_size, activation_str, ln):
        # input shape [k, P, H]
        # print(input.get_shape().as_list())
        # w shape [H, O]
        w = tf.get_variable(name + "_w_"+str(ln), [input_size, layer_size], dtype=self.dtype_tf, initializer=tf_initializers[self.initializer])
        # b shape [1, O]
        b = tf.get_variable(name + "_b_"+str(ln), [1, layer_size], dtype=self.dtype_tf, initializer=tf_initializers[self.initializer])
        o = tf.tensordot(input, w, [[2],[0]]) + b
        o = tf_activations[activation_str](o)
        # output shape [k, P, O]
        return o

    def getKeysInModelName(self):
        keys = {
        'num_parallel_groups':'NUM_PARALL_GROUPS', 
        'RDIM':'RDIM', 
        'N_used':'N_used', 
        'rnn_num_layers':'NUM-LAY', 
        'rnn_size_layers':'SIZE-LAY', 
        'rnn_activation_str':'ACT',
        'initial_state_handling':'ISH',
        'sequence_length':'SL',
        'prediction_length':'PL',
        'learning_rate':'LR',
        'dropout_keep_prob':'DKP',
        'zoneout_keep_prob':'ZKP',
        'hidden_state_propagation_length':'HSPL',
        'iterative_prediction_length':'IPL',
        'noise_level':'NL',
        #'num_test_ICS':'NICS',
        'num_parallel_groups':'NG', 
        'parallel_group_size':'GS', 
        'parallel_group_interaction_length':'GIL', 
        # 'worker_id':'WID', 
        }
        return keys

    def createModelName(self, params):
        keys = self.getKeysInModelName()
        str_ = "GPU-" * self.GPU + "RNN-" + self.rnn_cell_type + "-PARALLEL"
        for key in keys:
            str_ += "-" + keys[key] + "_{:}".format(params[key])
        return str_

    def __init__(self, params):

        if params["RDIM"] % params["num_parallel_groups"]: raise ValueError("ERROR: The num_parallel_groups should divide RDIM.")
        if size != params["num_parallel_groups"]: raise ValueError("ERROR: The num_parallel_groups is not equal to the number or ranks. Aborting...")
        
        self.display_output = params["display_output"]
        self.GPU = True if (tf.test.is_gpu_available()==True) else False

        # PARALLEL MODEL
        self.num_parallel_groups = params["num_parallel_groups"]
        self.RDIM = params["RDIM"]
        self.N_used = params["N_used"]
        self.dropout_keep_prob = params["dropout_keep_prob"]
        self.zoneout_keep_prob = params["zoneout_keep_prob"]
        
        self.parallel_group_interaction_length = params["parallel_group_interaction_length"]
        params["parallel_group_size"] = int(params["RDIM"]/params["num_parallel_groups"])
        self.parallel_group_size = params["parallel_group_size"]

        self.worker_id = rank
        self.parallel_group_num = rank

        self.input_dim = params["parallel_group_size"] + params["parallel_group_interaction_length"] * 2
        self.output_dim = params["parallel_group_size"]


        self.num_test_ICS = params["num_test_ICS"]
        self.iterative_prediction_length = params["iterative_prediction_length"]
        self.hidden_state_propagation_length = params["hidden_state_propagation_length"]

        self.rnn_activation_str = params['rnn_activation_str']
        self.initializer = params['initializer']

        self.initial_state_handling = "statefull"
        params["initial_state_handling"] = "statefull"

        self.rnn_num_layers = params['rnn_num_layers']
        self.rnn_size_layers = params['rnn_size_layers']

        self.sequence_length = params['sequence_length']
        self.rnn_cell_type = params['rnn_cell_type']

        self.scaler_tt = params["scaler"]
        self.scaler = scaler(self.scaler_tt)
        self.noise_level = params["noise_level"]

        self.regularization =  params['regularization']
        self.retrain =  params['retrain']
        self.subsample =  params['subsample']
        self.train_val_ratio = params['train_val_ratio']

        self.batch_size =  params['batch_size']
        self.overfitting_patience =  params['overfitting_patience']
        self.training_min_epochs =  params['training_min_epochs']
        self.max_epochs = params['max_epochs']
        self.num_rounds = params['num_rounds']
        self.learning_rate =  params['learning_rate']
        # Transforming the reference training time from hours to minutes
        # Half an hour buffer for saving the model
        self.reference_train_time = 60*60*(params["reference_train_time"]-params["buffer_train_time"])
        print("Reference train time {:} seconds / {:} minutes / {:} hours.".format(self.reference_train_time, self.reference_train_time/60, self.reference_train_time/60/60))

        if self.initializer not in ['xavier', 'normal']:
            raise ValueError('ERROR: INVALID INITIALIZER!')
        if self.rnn_activation_str not in ['tanh', 'sigmoid', 'relu'] and self.rnn_cell_type not in ['unitary']:
            raise ValueError('ERROR: INVALID RNN ACTIVATION!')

        self.dtype_tf = TFDTYPE
        self.dtype_np = NPDTYPE

        # ADDING COMPLEX DTYPES
        if self.rnn_cell_type == 'unitary':
            self.unitary_capacity = params["unitary_capacity"]
            if params["unitary_cplex"] == True:
                self.unitary_cplex = True
                self.cdtype_tf = TFCDTYPE
                self.cdtype_np = NPCDTYPE
            else:
                self.unitary_cplex = False
                self.cdtype_tf = TFDTYPE
                self.cdtype_np = NPDTYPE
        else:
            self.unitary_cplex = False
            self.cdtype_tf = TFDTYPE
            self.cdtype_np = NPDTYPE

        self.prediction_length = params['prediction_length']

        # rnn sequence input and target
        self.input = tf.placeholder(self.dtype_tf, shape=[None, None, self.input_dim])
        self.target = tf.placeholder(self.dtype_tf, shape=[None, None, self.output_dim])
        self._dropout_keep_prob = tf.placeholder(self.dtype_tf, shape=())
        self._zoneout_keep_prob = tf.placeholder(self.dtype_tf, shape=())
        self._is_training = tf.placeholder(tf.bool, shape=())
        
        self.initial_hidden_states = []
        for ln in range(self.rnn_num_layers):
            if self.rnn_cell_type == 'lstm':
                c_state = tf.placeholder(self.cdtype_tf, shape=[None, self.rnn_size_layers])
                h_state = tf.placeholder(self.cdtype_tf, shape=[None, self.rnn_size_layers])
                initial_state = tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state)
                self.initial_hidden_states.append(initial_state)
            else:
                self.initial_hidden_states.append(tf.placeholder(self.cdtype_tf, shape=[None, self.rnn_size_layers]))

        layer_input = self.input
        input_size = self.input_dim

        rnn_last_states = []
        rnn_outputs = []
        for ln in range(self.rnn_num_layers):
            layer_output, last_states = self.rnn_layer(self.rnn_cell_type, 'rnn', layer_input, input_size, self.rnn_size_layers, self.rnn_activation_str, ln, self.initial_hidden_states[ln])
            rnn_last_states.append(last_states)
            rnn_outputs.append(layer_input)
            # RESIDUAL CONNECTIONS IN CASE OF MULTIPLE LAYERS (AFTER THE FIRST LAYER WHERE THE INPUT HAS DIFFERENT DIMENSION)
            # layer_input = layer_output if ln==0 else layer_output + layer_input
            layer_input = tf.cast(layer_output, dtype=self.dtype_tf) if ln==0 else tf.cast(layer_output, dtype=self.dtype_tf) + layer_input
            input_size = self.rnn_size_layers

        self.rnn_last_states = rnn_last_states
        self.rnn_outputs = rnn_outputs

        # print(layer_output.get_shape().as_list())
        self.rnn_output = self.mlp_unfolded_layer("rnn_mlp_output", layer_output, input_size, self.output_dim, 'identity', ln)

        # Only the last self.prediction_length elements constitute to the loss function
        self.rnn_output_loss = self.rnn_output[:,-self.prediction_length:]

        # MSE Loss of the rnn
        self.rmse_loss = tf.reduce_mean(tf.squared_difference(self.rnn_output_loss, self.target))

        self.trainable_variables = tf.trainable_variables()

        if self.parallel_group_num==0:
            print("######################")
            print("TRAINABLE PARAMETERS:")
            for var in tf.trainable_variables():
                print(var.name)
            

        self.trainable_variables_names = [var.name for var in self.trainable_variables]
        self.n_model_parameters = computeNumberOfModelParameters(self.trainable_variables)
        self.n_trainable_parameters = self.n_model_parameters
        if self.parallel_group_num==0:
            print("Number of total parameters in RNN: {:d}".format(self.n_model_parameters))
            

        # INITIALIZING SESSION
        self.session = tf.Session()

        # SETTING THE PATHS...
        self.main_train_data_path = params['train_data_path']
        self.main_test_data_path = params['test_data_path']
        self.saving_path = params['saving_path']
        self.model_dir = params['model_dir']
        self.fig_dir = params['fig_dir']
        self.results_dir = params["results_dir"]
        self.logfile_dir = params["logfile_dir"]
        self.write_to_log = params["write_to_log"]
        self.model_name = self.createModelName(params)
        self.member_saving_model_path = self.saving_path + self.model_dir + self.model_name + "/model_{:d}".format(self.parallel_group_num)
        
        # MASTER RANK CREATING DIRECTORIES
        if rank==0:
            os.makedirs(self.saving_path + self.model_dir + self.model_name, exist_ok=True)
            os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
            os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
            os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)

    def regularizationLoss(self):
        if not isZeroOrNone(self.regularization):
            print("#### List of variables where regularization is applied: ####")
            vars_ = self.trainable_variables
            for var in vars_:
                if "bias" not in var.name and "_b_" not in var.name:
                    print(var.name)
                    print("###########################################")
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars_ if "bias" not in v.name and "_b_" not in v.name]) * self.regularization
            
            return lossL2
        else:
            return 0.0

    def defineLoss(self):
        with tf.name_scope('Losses'):
            loss = self.rmse_loss + self.regularizationLoss()
        return loss

    def clip_grad_norms(self, gradients_to_variables, max_norm=5):
        grads_and_vars = []
        for grad, var in gradients_to_variables:
            if grad is not None:
                if isinstance(grad, tf.IndexedSlices):
                    tmp = tf.clip_by_norm(grad.values, max_norm)
                    grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
                else:
                    grad = tf.clip_by_norm(grad, max_norm)
            grads_and_vars.append((grad, var))
        return grads_and_vars

    def defineTrainer(self, vars):
        with tf.name_scope('Optimizer_Scope'):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            # optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_round, beta1=0.9, beta2=0.999)
            # Op to calculate every variable gradient
            grads = tf.gradients(self.loss, vars)
            grads = list(zip(grads, vars))
            # grads = self.clip_grad_norms(grads, max_norm=0.001)
            # Op to update all variables according to their gradient
            trainer = optimizer.apply_gradients(grads_and_vars=grads)

        return trainer, grads, optimizer

    def plotBatchNumber(self, i, n_batches, train):
        if self.display_output == True and self.parallel_group_num == 0:
            str_ = train * "TRAINING: " + (not train)* "EVALUATION"
            print("{:s} batch {:d}/{:d},  {:f}%".format(str_, int(i+1), int(n_batches), (i+1)/n_batches*100.))
            sys.stdout.write("\033[F")
            

    def getBatch(self, sequence, batch_idx):
        input_batch = []
        target_batch = []
        for predict_on in batch_idx:
            input = sequence[predict_on-self.sequence_length:predict_on+self.prediction_length-1]
            target = sequence[predict_on:predict_on+self.prediction_length, getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)]
            target = np.reshape(target, (self.prediction_length, -1))
            input_batch.append(input)
            target_batch.append(target)
        input_batch = np.array(input_batch)
        target_batch = np.array(target_batch)
        return input_batch, target_batch

    def getZeroStates(self, num_layers, size_layer, batch_size):
        initial_states = []
        for ln in range(num_layers):
            if self.rnn_cell_type =="lstm":
                initial_states.append(tuple([np.zeros((batch_size, size_layer), dtype=self.dtype_np), np.zeros((batch_size, size_layer), dtype=self.dtype_np)]))
            else:
                initial_states.append(np.zeros((batch_size, size_layer), dtype=self.dtype_np))
        return initial_states

    def trainOnBatch(self, idx_train_on_epoch, batch_idx, input_sequence, train=False):
        idx_train_on_epoch -= set(batch_idx)
        temp = np.array([range(j-self.sequence_length, j) for j in batch_idx]).flatten()
        idx_train_on_epoch -= set(temp)
        temp = np.array([range(j, j+self.hidden_state_propagation_length+self.prediction_length) for j in batch_idx]).flatten()
        idx_train_on_epoch -= set(temp)
        initial_hidden_states = self.getZeroStates(self.rnn_num_layers, self.rnn_size_layers, np.shape(batch_idx)[0])
        rnn_loss_all = []
        for p in range(int(self.hidden_state_propagation_length//(self.sequence_length+self.prediction_length))):
            input_batch, target_batch = self.getBatch(input_sequence, batch_idx)
            feed_dict = {self.input:input_batch, self.target:target_batch}
            hidden_state_dict = {i: d for i, d in zip(self.initial_hidden_states, initial_hidden_states)}
            feed_dict.update(hidden_state_dict)
            if train == False:
                feed_dict.update({self._dropout_keep_prob:1.0, self._zoneout_keep_prob:1.0, self._is_training:False, })
                rnn_loss, last_states = self.session.run([self.rmse_loss, self.rnn_last_states], feed_dict=feed_dict)
            else:
                feed_dict.update({self._dropout_keep_prob:self.dropout_keep_prob, self._zoneout_keep_prob:self.zoneout_keep_prob, self._is_training:True })
                rnn_loss, last_states, _ = self.session.run([self.rmse_loss, self.rnn_last_states, self.trainer_rnn], feed_dict=feed_dict)
            rnn_loss_all.append(rnn_loss)
            initial_hidden_states = last_states
            batch_idx = np.array(batch_idx) + self.sequence_length + self.prediction_length
        rnn_loss = np.mean(np.array(rnn_loss_all))
        return idx_train_on_epoch, rnn_loss



    def trainEpoch(self, idx_on, n_samples, input_sequence, train=False):
        idx_on_epoch = idx_on.copy()
        epoch_loss_all = []
        stop_limit = np.max([self.hidden_state_propagation_length, self.batch_size])
        stop_limit = np.min([stop_limit, len(idx_on)])
        while len(idx_on_epoch) >= stop_limit:
            self.plotBatchNumber(n_samples-len(idx_on_epoch), n_samples, train)
            batch_idx = random.sample(idx_on_epoch, self.batch_size)
            idx_on_epoch, rnn_loss = self.trainOnBatch(idx_on_epoch, batch_idx, input_sequence, train=train)
            # print("NUMBER OF REMAINING SAMPLES {:d}".format(len(idx_on_epoch)))
            epoch_loss_all.append(rnn_loss)
        epoch_loss = np.mean(np.array(epoch_loss_all))
        return epoch_loss

    def getStartingPoints(self, input_sequence):
        NN = np.shape(input_sequence)[0]
        if NN - self.hidden_state_propagation_length - self.sequence_length < 0:
            raise ValueError("The hidden_state_propagation_length is too big. Reduce it. N_data !> H + SL, {:} !> {:} + {:} = {:}".format(NN, self.hidden_state_propagation_length, self.sequence_length, self.sequence_length+self.hidden_state_propagation_length))
        idx_on = set(np.arange(self.sequence_length, NN - self.hidden_state_propagation_length))
        n_samples = len(idx_on)
        return idx_on, n_samples

    # def train_signal_handler(self, sig, frame):
    #     comm.Barrier()
    #     print('\nSignal catched: {:} by rank {:}'.format(sig, self.parallel_group_num))
    #     print("Rank {:} saving model.".format(self.parallel_group_num))
    #     self.saveModel()
    #     plotTrainingLosses(self, self.rnn_loss_train_vec, self.rnn_loss_val_vec, self.rnn_min_val_error, "_member_{:d}".format(self.parallel_group_num))
    #     comm.Barrier()
    #     # sys.exit(0)

    def train(self):
        self.start_time = time.time()
        # signal.signal(signal.SIGUSR2, self.train_signal_handler)
        # signal.signal(signal.SIGUSR1, self.train_signal_handler)
        # signal.signal(signal.SIGINT, self.train_signal_handler)
        print('My PID is:', os.getpid())
        # time.sleep(30000)
        

        self.worker_train_data_path, self.worker_test_data_path = createParallelTrainingData(self)
        with open(self.worker_train_data_path, "rb") as file:
            data = pickle.load(file)
            input_sequence = data["train_input_sequence"][:, :self.input_dim]
            print("Adding noise to the training data. {:} per mille ".format(self.noise_level))
            input_sequence = addNoise(input_sequence, self.noise_level)
            N_all, dim = np.shape(input_sequence)
            if self.input_dim > dim: raise ValueError("Requested input dimension is wrong.")
            input_sequence = input_sequence[:self.N_used, :self.input_dim]
            dt = data["dt"]
            del data
            
        if self.parallel_group_num==0: print("##Using {:}/{:} dimensions and {:}/{:} samples ##".format(self.input_dim, dim, self.N_used, N_all))
        if self.N_used > N_all: raise ValueError("Not enough samples in the training data.")
        

        if self.parallel_group_num==0: print("SCALING")
        input_sequence = self.scaler.scaleData(input_sequence)

        print(input_sequence.dtype)
        print(np.shape(input_sequence))

        T = int(np.shape(input_sequence)[0]//self.subsample)
        input_sequence = input_sequence[:T]

        train_input_sequence, val_input_sequence = divideData(input_sequence, self.train_val_ratio)

        idx_train_on, n_train_samples = self.getStartingPoints(train_input_sequence)
        idx_val_on, n_val_samples = self.getStartingPoints(val_input_sequence)

        if self.parallel_group_num==0: print("NUMBER OF TRAINING SAMPLES: {:d}".format(n_train_samples))
        if self.parallel_group_num==0: print("NUMBER OF VALIDATION SAMPLES: {:d}".format(n_val_samples))

        self.rnn_loss_train_vec = []
        self.rnn_loss_val_vec = []
        for round_num in range(self.num_rounds):
            isWallTimeLimit = self.trainRound(round_num, idx_train_on, idx_val_on, n_train_samples, n_val_samples, train_input_sequence, val_input_sequence)
            if isWallTimeLimit: break

        # If the training time limit was not reached, save the model...
        if not isWallTimeLimit:
            print("## Training converged: Rank {:} saving model... ##".format(self.parallel_group_num))
            self.saveModel()
            plotTrainingLosses(self, self.rnn_loss_train_vec, self.rnn_loss_val_vec, self.rnn_min_val_error, "_member_{:d}".format(self.parallel_group_num))

        # SYNCHRONIZING
        comm.Barrier()
        print("SYNCHRONIZING...")


    def trainRound(self, round_num, idx_train_on, idx_val_on, n_train_samples, n_val_samples, train_input_sequence, val_input_sequence):
        # Check if retraining of a model is requested else random initialization of the weights
        self.loss = self.defineLoss()
        self.learning_rate_round = self.learning_rate/(pow(10, round_num))
        self.trainer_rnn, self.grads, self.optimizer = self.defineTrainer(self.trainable_variables)
        # self.trainer_reset = tf.variables_initializer(self.optimizer.variables())
        saver = tf.train.Saver(max_to_keep=100000, var_list=self.trainable_variables)

        # # DEFINE REINITIALIZATION FOR OPTIMIZER
        # var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # reset_opt_op = tf.variables_initializer([self.optimizer.get_slot(var, name) for name in self.optimizer.get_slot_names() for var in var_list])

        optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 "Optimizer_Scope")
        reset_opt_op = tf.variables_initializer(optimizer_scope)


        if self.retrain == 1 or round_num>0:
            print("!! RESETTING THE OPTIMIZER !!")
            saver.restore(self.session, self.member_saving_model_path)
            # self.session.run(tf.global_variables_initializer())
            self.session.run(reset_opt_op)
            self.trainer_reset = tf.variables_initializer(self.optimizer.variables())
            self.session.run(self.trainer_reset)
        else:
            self.session.run(tf.global_variables_initializer())
            # SAVING THE INITIAL MODEL
            saver.save(self.session, self.member_saving_model_path)

        print("##### ROUND: {:}, LEARNING RATE={:} #####".format(round_num, self.learning_rate_round))
        

        rnn_loss_train = self.trainEpoch(idx_train_on, n_train_samples, train_input_sequence, train=False)
        rnn_loss_val = self.trainEpoch(idx_val_on, n_val_samples, val_input_sequence, train=False)

        print("INITIAL TRAIN rnn_loss: ", rnn_loss_train)
        print("INITIAL VAL rnn_loss: ", rnn_loss_val)
        

        self.rnn_min_val_error = rnn_loss_val
        self.rnn_train_error = rnn_loss_train

        rnn_loss_val_round_vec = []
        self.rnn_loss_train_vec.append(rnn_loss_train)
        rnn_loss_val_round_vec.append(rnn_loss_val)
        self.rnn_loss_val_vec.append(rnn_loss_val)
        for epoch in range(self.max_epochs):
            rnn_loss_train = self.trainEpoch(idx_train_on, n_train_samples, train_input_sequence, train=True)
            rnn_loss_val = self.trainEpoch(idx_val_on, n_val_samples, val_input_sequence, train=False)
            self.rnn_loss_train_vec.append(rnn_loss_train)
            rnn_loss_val_round_vec.append(rnn_loss_val)
            self.rnn_loss_val_vec.append(rnn_loss_val)
            if epoch%1 == 0:
                print("########################################################")
                print("ROUND {:d} EPOCH {:d}".format(round_num, epoch))
                print("TRAIN loss: ", rnn_loss_train)
                print("VAL loss: ", rnn_loss_val)
                print("\n")
                
            if rnn_loss_val < self.rnn_min_val_error:
                self.rnn_min_val_error = rnn_loss_val
                self.rnn_train_error = rnn_loss_train
                saver.save(self.session, self.member_saving_model_path)
            if epoch > self.overfitting_patience + self.training_min_epochs:
                if all(self.rnn_min_val_error < rnn_loss_val_round_vec[-self.overfitting_patience:]):
                    break
            isWallTimeLimit = self.isWallTimeLimit()
            if isWallTimeLimit: break
        return isWallTimeLimit

    def isWallTimeLimit(self):
        training_time = time.time() - self.start_time
        if training_time > self.reference_train_time:
            print("## Maximum train time reached: Rank {:} saving model... ##".format(self.parallel_group_num))
            self.saveModel()
            plotTrainingLosses(self, self.rnn_loss_train_vec, self.rnn_loss_val_vec, self.rnn_min_val_error, "_member_{:d}".format(self.parallel_group_num))
            return True
        else:
            return False

    def saveModel(self):
        if self.parallel_group_num==0: print("Recording time...")
        self.total_training_time = time.time() - self.start_time
        if hasattr(self, 'rnn_loss_train_vec'):
            if len(self.rnn_loss_train_vec)!=0:
                # Tracking the training time per epoch
                self.training_time = self.total_training_time/len(self.rnn_loss_train_vec)
        else:
            self.training_time = self.total_training_time

        if self.parallel_group_num==0: print("Total training time per epoch is {:}".format(self.training_time))
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
        "total_training_time":self.total_training_time,
        "training_time":self.training_time,
        "n_trainable_parameters":self.n_trainable_parameters,
        "n_model_parameters":self.n_model_parameters,
        "rnn_loss_train_vec":self.rnn_loss_train_vec,
        "rnn_loss_val_vec":self.rnn_loss_val_vec,
        "rnn_min_val_error":self.rnn_min_val_error,
        "rnn_train_error":self.rnn_train_error,
        "scaler":self.scaler,
        }
        data_path = self.saving_path + self.model_dir + self.model_name + "/data_{:d}.pickle".format(self.parallel_group_num)
        with open(data_path, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
        print("## RANK {:} SAVED THE MODEL! ##".format(self.parallel_group_num))
        
    def loadModel(self):
        saver = tf.train.Saver(max_to_keep=100000, var_list=self.trainable_variables)
        self.session.run(tf.global_variables_initializer())

        try:
            saver.restore(self.session, self.member_saving_model_path)
        except:
            print("MODEL {:s} NOT FOUND.".format(self.member_saving_model_path))
            return 1

        data_path = self.saving_path + self.model_dir + self.model_name + "/data_{:d}.pickle".format(self.parallel_group_num)
        with open(data_path, "rb") as file:
            data = pickle.load(file)
            self.scaler = data["scaler"]
            del data
        return 0

    def testing(self):
        self.worker_train_data_path, self.worker_test_data_path = createParallelTrainingData(self)
        if self.loadModel()==0:
            self.n_warmup = int(self.hidden_state_propagation_length/self.sequence_length//4)
            if self.parallel_group_num==0: print("WARMING UP STEPS (for statefull RNNs): {:d}".format(self.n_warmup))
            if self.parallel_group_num==0: print("# TEST ON TRAINING SET #")
            self.testingOnTrainingSet()
            if self.parallel_group_num==0: print("# TEST ON TESTING SET #")
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
            if self.parallel_group_num == 0:
                print("IC {:}/{:}, {:2.3f}%".format(ic_num, num_test_ICS, ic_num/num_test_ICS*100))
            ic_idx = ic_indexes[ic_num]
            input_sequence_ic = input_sequence[ic_idx-self.sequence_length-self.n_warmup:ic_idx+self.iterative_prediction_length]
            prediction, target, _, _ = self.predictSequence(input_sequence_ic)
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

    def predictSequence(self, input_sequence):

        N = np.shape(input_sequence)[0]
        # PREDICTION LENGTH
        if N - self.sequence_length - self.n_warmup != self.iterative_prediction_length: raise ValueError("Error! N - self.sequence_length - self.n_warmup != iterative_prediction_length")

        # PREPARING THE HIDDEN STATES
        zero_ihs = []
        # initial_hidden_states size is [nl, nb, bs, nh]
        for i in range(self.rnn_num_layers):
            if self.rnn_cell_type == "lstm":
                zero_ihs.append(tuple([np.zeros((1, self.rnn_size_layers)), np.zeros((1, self.rnn_size_layers))]))
            else:
                # initial_hidden_state size is [nb, bs, nh]
                zero_ihs.append(np.zeros((1, self.rnn_size_layers)))

        warmup_data_input = np.reshape(input_sequence[:self.sequence_length + self.n_warmup-1], (1, -1, self.input_dim))
        warmup_data_target = np.reshape(input_sequence[1:self.sequence_length + self.n_warmup, getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)], (1, -1, self.output_dim))
        # print(np.shape(warmup_data_input))
        # print(np.shape(warmup_data_target))

        target = input_sequence[self.sequence_length+self.n_warmup:, getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)]
        input_t = np.reshape(input_sequence[self.sequence_length+self.n_warmup-1], (1, -1, self.input_dim))

        # print(np.shape(target))
        # print(np.shape(input_t))
        # print(ark)

        # Initializing zero hidden state
        hidden_state_dict = {i: d for i, d in zip(self.initial_hidden_states, zero_ihs)}
        feed_dict = dict({self.input:warmup_data_input, self._dropout_keep_prob:1.0, self._zoneout_keep_prob:1.0, self._is_training:False, })
        feed_dict.update(hidden_state_dict)

        warmup_data_output, last_states = self.session.run([self.rnn_output, self.rnn_last_states], feed_dict=feed_dict)
        # print(np.shape(warmup_data_output))
        # print(np.shape(last_states))

        # print(ark)
        prediction = []
        for t in range(self.iterative_prediction_length):
            if self.display_output and self.parallel_group_num == 0: print("PREDICTION: T {:}/{:}, {:2.3f}%".format(t, self.iterative_prediction_length, t/(self.iterative_prediction_length*100), end="\r"))
            hidden_state_dict = {i: d for i, d in zip(self.initial_hidden_states, last_states)}
            feed_dict = dict({self.input:input_t, self._dropout_keep_prob:1.0, self._zoneout_keep_prob:1.0, self._is_training:False, })
            feed_dict.update(hidden_state_dict)
            data_output, last_states = self.session.run([self.rnn_output, self.rnn_last_states], feed_dict=feed_dict)
            # print(np.shape(data_output))
            # priont(ark)
            prediction.append(data_output[0,0,:])
            # LOCAL STATE
            global_state = np.zeros((self.RDIM))
            local_state = np.zeros((self.RDIM))
            local_state[self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = data_output[0,0,:].copy()
            comm.Barrier()
            # UPDATING THE GLOBAL STATE - IMPLICIT BARRIER
            comm.Allreduce([local_state, MPI.DOUBLE], [global_state, MPI.DOUBLE], MPI.SUM)
            state_list = Circ(list(global_state.copy()))
            group_start = self.parallel_group_num * self.parallel_group_size
            group_end = group_start + self.parallel_group_size
            pgil = self.parallel_group_interaction_length
            input_t = []
            for i in range(group_start-pgil, group_end+pgil):
                input_t.append(state_list[i].copy())
            input_t = np.array(input_t)
            input_t = np.reshape(input_t, (1, -1, self.input_dim))

        target_augment = np.concatenate((warmup_data_target[0], target), axis=0)
        prediction_augment = np.concatenate((warmup_data_output[0], prediction), axis=0)
        return prediction, target, prediction_augment, target_augment
                        
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
        # SAVING RESULTS ON FILE
        return 0
