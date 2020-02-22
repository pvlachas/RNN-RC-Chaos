#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import sys
import os
from plotting_utils import *
from global_utils import *
import pickle
import time
import signal
import tensorflow as tf

# MEMORY TRACKING
import psutil

TFDTYPE = tf.float64
NPDTYPE = np.float64

# TFDTYPE = tf.float32
# NPDTYPE = np.float32

tf_activations = {"relu": tf.nn.relu, "tanh": tf.tanh, "sigmoid": tf.sigmoid, "identity": tf.identity, "softplus":tf.nn.softplus, "softmax":tf.nn.softmax}
tf_initializers = {"xavier": tf.contrib.layers.xavier_initializer(), "normal": tf.truncated_normal_initializer()}

class mlp(object):
    def delete(self):
        print("Resetting default graph!")
        tf.reset_default_graph()

    def mlp_layer(self, name, input, input_size, layer_size, activation_str, ln):
        # input has shape [None, num_hidden]
        w = tf.get_variable(name + "_w_"+str(ln), [input_size, layer_size], dtype=TFDTYPE, initializer=tf_initializers[self.initializer])
        b = tf.get_variable(name + "_b_"+str(ln), [1, layer_size], dtype=TFDTYPE, initializer=tf_initializers[self.initializer])
        o = tf_activations[activation_str](tf.einsum('kj,jh->kh', input, w) + b)
        return o

    def mlp_input_layer(self, name, input, input_size, sequence_length, layer_size, activation_str, ln):
        # input has shape [None, sequence_length, num_hidden]
        # layer_size = input_dim (MLP has to output the input sequence)
        w = tf.get_variable(name + "_w_"+str(ln), [sequence_length, input_size, layer_size], dtype=TFDTYPE, initializer=tf_initializers[self.initializer])
        b = tf.get_variable(name + "_b_"+str(ln), [1, layer_size], dtype=TFDTYPE, initializer=tf_initializers[self.initializer])
        o = tf_activations[activation_str](tf.einsum('ksj,sjh->kh', input, w) + b)
        return o
        
    def getKeysInModelName(self):
        keys = {
        'RDIM':'RDIM', 
        'N_used':'N_used', 
        'mlp_num_layers':'NUM-LAY', 
        'mlp_size_layers':'SIZE-LAY', 
        'mlp_activation_str':'ACT', 
        'sequence_length':'SL',
        'learning_rate':'LR',
        'regularization':'REG',
        'iterative_prediction_length':'IPL',
        #'num_test_ICS':'NICS', 
        'worker_id':'WID', 
        }
        return keys

    def createModelName(self, params):
        keys = self.getKeysInModelName()
        str_ = "GPU-" * self.GPU + "MLP"
        for key in keys:
            str_ += "-" + keys[key] + "_{:}".format(params[key])
        return str_

    def __init__(self, params):
        self.display_output = params["display_output"]
        self.GPU = True if (tf.test.is_gpu_available()==True) else False


        self.regularization = params['regularization']
        self.retrain = params['retrain']
        self.subsample = params['subsample']
        self.train_val_ratio = params['train_val_ratio']
        self.batched_valtrain = params['batched_valtrain']
        self.num_rounds = params['num_rounds']

        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.overfitting_patience = params['overfitting_patience']
        self.training_min_epochs = params['training_min_epochs']
        self.max_epochs = params['max_epochs']
        self.scaler_tt = params["scaler"]
        self.scaler = scaler(self.scaler_tt)
        self.keep_prob = params["keep_prob"]
        self.num_test_ICS = params["num_test_ICS"]
        self.iterative_prediction_length = params["iterative_prediction_length"]

        self.initializer = params['initializer']

        self.mlp_num_layers = params['mlp_num_layers']
        self.mlp_size_layers = params['mlp_size_layers']
        self.mlp_activation_str = params['mlp_activation_str']

        self.input_dim = params['RDIM']
        self.output_dim = params["RDIM"]
        self.N_used = params["N_used"]

        if self.initializer not in ['xavier', 'normal']:
            raise ValueError('ERROR: INVALID INITIALIZER!')
        if self.mlp_activation_str not in ['tanh', 'sigmoid', 'relu', 'identity']:
            raise ValueError('ERROR: INVALID MLP ACTIVATION!')

        # setting the prediction length
        if params['prediction_length']!= 1:
            raise ValueError('ERROR: prediction_length bigger than one. Not allowed. NOT ALLOWED!')

        self.dtype_tf = TFDTYPE
        self.dtype_np = NPDTYPE

        self.prediction_length = params['prediction_length']
        self.sequence_length = params['sequence_length']

        self.mlp_activation = tf_activations[self.mlp_activation_str]

        self.dropout_keep_prob = tf.placeholder(TFDTYPE, shape=())

        # MLP sequence input and target
        self.input = tf.placeholder(TFDTYPE, shape=[None, self.sequence_length, self.input_dim])
        self.target = tf.placeholder(TFDTYPE, shape=[None, self.prediction_length, self.output_dim])


        layer_input = tf.reshape(self.input, (tf.shape(self.input)[0], -1))
        input_size = self.input_dim * self.sequence_length

        for ln in range(self.mlp_num_layers):
            layer_output = self.mlp_layer('mlp', layer_input, input_size, self.mlp_size_layers, self.mlp_activation_str, ln)

            if ln > 0:
                layer_input = layer_output + layer_input
            else:
                layer_input = layer_output

            layer_input = tf.nn.dropout(layer_input, self.dropout_keep_prob)
            input_size = self.mlp_size_layers

        if self.mlp_num_layers == 0:
            self.mlp_output = self.mlp_layer("mlp_output", layer_input, input_size, self.output_dim, 'identity', 0)
        else:
            self.mlp_output = self.mlp_layer("mlp_output", layer_input, self.mlp_size_layers, self.output_dim, 'identity', ln)
        self.mlp_output = tf.reshape(self.mlp_output, (tf.shape(self.mlp_output)[0], 1, tf.shape(self.mlp_output)[1]))


        # MSE Loss of the MLP
        self.rmse_loss = tf.reduce_mean(tf.squared_difference(self.mlp_output, self.target))


        self.trainable_variables = tf.trainable_variables()

        print("######################")
        print("TRAINABLE PARAMETERS:")
        for var in tf.trainable_variables():
            print(var.name)
        print("######################")

        self.trainable_variables_names = [var.name for var in self.trainable_variables]
        self.n_model_parameters = computeNumberOfModelParameters(self.trainable_variables)
        print("Number of total parameters in MLP: {:d}".format(self.n_model_parameters))
        self.n_trainable_parameters = self.n_model_parameters
        self.session = tf.Session()

        # SETTING THE PATHS...
        self.train_data_path = params['train_data_path']
        self.test_data_path = params['test_data_path']
        self.saving_path = params['saving_path']
        self.model_dir = params['model_dir']
        self.fig_dir = params['fig_dir']
        self.results_dir = params['results_dir']
        self.logfile_dir = params["logfile_dir"]
        self.write_to_log = params["write_to_log"]
        self.model_name = self.createModelName(params)
        self.saving_model_path = self.saving_path + self.model_dir + self.model_name + "/"
        
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
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999)
            # Op to calculate every variable gradient
            grads = tf.gradients(self.loss, vars)
            grads = list(zip(grads, vars))
            # grads = self.clip_grad_norms(grads, max_norm=0.001)
            # Op to update all variables according to their gradient
            trainer = optimizer.apply_gradients(grads_and_vars=grads)
        return trainer, grads, optimizer

    def plotBatchNumber(self, i, n_batches):
        if self.display_output == True:
            print("batch {:d}/{:d},  {:f}%".format(int(i+1), int(n_batches), (i+1)/n_batches*100.))
            sys.stdout.write("\033[F")


    def evaluateLossAndTrain(self, sess, batched_valtrain, input_batches, target_batches, kp):
        loss = []
        n_batches = getFirstDataDimension(input_batches)
        for i in range(n_batches-1):
            self.plotBatchNumber(i, n_batches)
            batch_loss = self.runLossAndTrain(sess, kp, np.array(input_batches[i]), np.array(target_batches[i]))
            loss.append(batch_loss)
        loss = np.array(loss).mean()
        return loss

    def evaluateLoss(self, sess, batched_valtrain, input_batches, target_batches, kp):
        if isZeroOrNone(batched_valtrain):
            target_batches = np.array(target_batches).reshape(-1, *target_batches.shape[-2:])
            input_batches = np.array(input_batches).reshape(-1, *input_batches.shape[-2:])
            loss = self.runLoss(sess, kp, input_batches, target_batches)
        else:
            loss = []
            n_batches = getFirstDataDimension(input_batches)
            for i in range(n_batches-1):
                self.plotBatchNumber(i, n_batches)
                batch_loss = self.runLoss(sess, kp, np.array(input_batches[i]), np.array(target_batches[i]))
                loss.append(batch_loss)
            loss = np.array(loss).mean()
        return loss

    def runLoss(self, sess, kp, input_data, target_data):
        feed_dict = dict({self.dropout_keep_prob:kp, self.input:input_data, self.target:target_data})
        loss = sess.run(self.loss, feed_dict=feed_dict)
        return loss

    def runLossAndTrain(self, sess, kp, input_data, target_data):
        feed_dict = dict({self.dropout_keep_prob:kp, self.input:input_data, self.target:target_data})
        loss, _ = sess.run([self.loss, self.trainer_mlp], feed_dict=feed_dict)
        return loss

    def train_signal_handler(self, sig, frame):
        print('\nSignal catched: {}'.format(sig))
        sys.stdout.flush()
        self.saveModel()
        plotTrainingLosses(self, self.loss_train_vec, self.loss_val_vec, self.min_val_error)
        sys.exit(0)

    def train(self):
        self.start_time = time.time()
        signal.signal(signal.SIGUSR2, self.train_signal_handler)
        signal.signal(signal.SIGINT, self.train_signal_handler)
        try:

            with open(self.train_data_path, "rb") as file:
                # Pickle the "data" dictionary using the highest protocol available.
                data = pickle.load(file)
                input_sequence = data["train_input_sequence"]
                N_all, dim = np.shape(input_sequence)
                if self.input_dim > dim: raise ValueError("Requested input dimension is wrong.")
                input_sequence = input_sequence[:self.N_used, :self.input_dim]
                dt = data["dt"]
                del data


            print("SCALING")
            input_sequence = self.scaler.scaleData(input_sequence)


            input_data, target_data = stackSequenceData(input_sequence, self.sequence_length, self.prediction_length, self.subsample)
            del input_sequence
            
            print(np.shape(input_data))
            print(np.shape(target_data))

            rem = np.shape(input_data)[0] % self.batch_size
            input_data = input_data[:-rem]
            target_data = target_data[:-rem]

            print(np.shape(input_data))
            print(np.shape(target_data))

            input_data = np.split(np.array(input_data), self.batch_size, axis=0)
            target_data = np.split(np.array(target_data), self.batch_size, axis=0)

            print(np.shape(input_data))
            print(np.shape(target_data))

            input_data = np.swapaxes(input_data, 0, 1)
            target_data = np.swapaxes(target_data, 0, 1)

            print(np.shape(input_data))
            print(np.shape(target_data))

            input_data, target_data = shuffle(input_data, target_data)
            print(np.shape(input_data))
            print(np.shape(target_data))


            # print("Input data shape: ", input_data.shape)
            # print("Target data shape: ", target_data.shape)
            # Divide input data in training and validation data set

            input_train_batches, input_val_batches = divideData(input_data, self.train_val_ratio)
            target_train_batches, target_val_batches = divideData(target_data, self.train_val_ratio)

            del input_data
            del target_data

            n_batches_train = np.shape(input_train_batches)[0]
            n_batches_val = np.shape(input_val_batches)[0]
            print("NUMBER OF TRAINING BATCHES: {:d}".format(n_batches_train))
            print("NUMBER OF VALIDATION BATCHES: {:d}".format(n_batches_val))
            print(np.shape(input_train_batches))
            print(np.shape(input_val_batches))

            self.loss_train_vec = []
            self.loss_val_vec = []
            for round_num in range(self.num_rounds):
                loss_train_vec, loss_val_vec = self.trainRound(round_num,n_batches_train, n_batches_val, input_train_batches, input_val_batches, target_train_batches, target_val_batches)
                self.loss_train_vec += loss_train_vec
                self.loss_val_vec += loss_val_vec

            self.saveModel()
            plotTrainingLosses(self, self.loss_train_vec, self.loss_val_vec, self.min_val_error)
        except Exception as e:
            print("Exception caught: {}".format(e))
            print("Saving model")
            sys.stdout.flush()
            self.saveModel()
            plotTrainingLosses(self, self.loss_train_vec, self.loss_val_vec, self.min_val_error)

    def trainRound(self, round_num,n_batches_train, n_batches_val, input_train_batches, input_val_batches, target_train_batches, target_val_batches):

        self.loss = self.defineLoss()
        self.learning_rate_round = self.learning_rate/(pow(10, round_num))
        self.trainer_mlp, self.grads, self.optimizer = self.defineTrainer(self.trainable_variables)
        # Training the MLP
        saver = tf.train.Saver(max_to_keep=100000, var_list=self.trainable_variables)

        # # DEFINE REINITIALIZATION FOR OPTIMIZER
        # var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # reset_opt_op = tf.variables_initializer([self.optimizer.get_slot(var, name) for name in self.optimizer.get_slot_names() for var in var_list])

        optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 "Optimizer_Scope")
        reset_opt_op = tf.initialize_variables(optimizer_scope)

        if self.retrain == 1 or round_num>0:
            print("!! RESETTING THE OPTIMIZER !!")
            saver.restore(self.session, self.saving_model_path)
            # self.session.run(tf.global_variables_initializer())
            self.session.run(reset_opt_op)
            self.trainer_reset = tf.variables_initializer(self.optimizer.variables())
            self.session.run(self.trainer_reset)
        else:
            self.session.run(tf.global_variables_initializer())
            # SAVING THE INITIAL MODEL
            saver.save(self.session, self.saving_model_path)

        print("##### ROUND: {:}, LEARNING RATE={:} #####".format(round_num, self.learning_rate_round))


        loss_train = self.evaluateLoss(self.session, self.batched_valtrain, input_train_batches, target_train_batches, 1.0)
        loss_val = self.evaluateLoss(self.session, self.batched_valtrain, input_val_batches, target_val_batches, 1.0)
        print("INITIAL TRAIN loss: ", loss_train)
        print("INITIAL VAL loss: ", loss_val)

        self.min_val_error = loss_val
        self.train_error = loss_train

        loss_train_vec = []
        loss_val_vec = []
        loss_train_vec.append(loss_train)
        loss_val_vec.append(loss_val)
        for epoch in range(self.max_epochs):
            loss_train = self.evaluateLossAndTrain(self.session, self.batched_valtrain, input_train_batches, target_train_batches, self.keep_prob)
            loss_val  = self.evaluateLoss(self.session, self.batched_valtrain, input_val_batches, target_val_batches, 1.0)

            loss_train_vec.append(loss_train)
            loss_val_vec.append(loss_val)

            if loss_val < self.min_val_error:
                self.min_val_error = loss_val
                self.train_error = loss_train
                saver.save(self.session, self.saving_model_path)

            if epoch > self.overfitting_patience + self.training_min_epochs:
                if all(self.min_val_error < loss_val_vec[-self.overfitting_patience:]):
                    break

            if epoch%1 == 0:
                print("########################################################")
                print("EPOCH {:d}".format(epoch))
                print("TRAIN loss: ", loss_train)
                print("VAL loss: ", loss_val)
                print("\n")
        return loss_train_vec, loss_val_vec


    def saveModel(self):
        print("Recording time...")
        self.total_training_time = time.time() - self.start_time
        self.training_time = self.total_training_time/len(self.loss_train_vec)
        print("Total training time per epoch is {:}".format(self.training_time))
        print("Total training time is {:}".format(self.total_training_time))

        print("MEMORY TRACKING IN MB...")
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss/1024/1024
        self.memory = memory
        print("Script used {:} MB".format(self.memory))

        if self.write_to_log == 1:
            logfile_train= self.saving_path + self.logfile_dir + self.model_name  + "/train.txt"
            writeToTrainLogFile(logfile_train, self)

        data = {
        "memory":self.memory,
        "total_training_time":self.total_training_time,
        "training_time":self.training_time,
        "n_model_parameters":self.n_model_parameters,
        "n_trainable_parameters":self.n_trainable_parameters,
        "loss_train_vec":self.loss_train_vec,
        "loss_val_vec":self.loss_val_vec,
        "min_val_error":self.min_val_error,
        "min_val_error":self.min_val_error,
        "train_error":self.train_error,
        "scaler":self.scaler,
        }
        data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
        with open(data_path, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data

    def loadModel(self):
        # Training the MLP
        saver = tf.train.Saver(max_to_keep=100000, var_list=self.trainable_variables)
        self.session.run(tf.global_variables_initializer())

        try:
            saver.restore(self.session, self.saving_model_path)
        except:
            print("MODEL {:s} NOT FOUND.".format(self.saving_model_path))
            return 1

        data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
        with open(data_path, "rb") as file:
            data = pickle.load(file)
            self.scaler = data["scaler"]
            del data
        return 0

    def testing(self):
        if self.loadModel() == 0:
            self.testingOnTrainingSet()
            self.testingOnTestingSet()
            self.saveResults()

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
            print("IC {:}/{:}, {:2.3f}%".format(ic_num, num_test_ICS, ic_num/num_test_ICS*100))
            ic_idx = ic_indexes[ic_num]
            input_sequence_ic = input_sequence[ic_idx-self.sequence_length:ic_idx+self.iterative_prediction_length]
            prediction, target = self.predictSequence(input_sequence_ic)
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
            if ic_num < 3: plotIterativePrediction(self, set_name, target, prediction, rmse, rmnse, ic_idx, dt)

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


    def predictSequence(self, input_sequence):
        N = np.shape(input_sequence)[0]
        # PREDICTION LENGTH
        if N - self.sequence_length != self.iterative_prediction_length: raise ValueError("Error! N - sequence_length != iterative_prediction_length")

        input_history = input_sequence[:self.sequence_length]
        input_history = np.reshape(input_history, (1, self.sequence_length, self.input_dim))

        target = input_sequence[self.sequence_length:]
        prediction = []
        for t in range(self.iterative_prediction_length):
            print("PREDICTION: T {:}/{:}, {:2.3f}%".format(t, self.iterative_prediction_length, t/self.iterative_prediction_length*100), end="\r")
            out = self.session.run(self.mlp_output, feed_dict={self.input:input_history, self.dropout_keep_prob:1.0})
            input_history = np.concatenate((input_history, out), axis=1)
            input_history = np.reshape(input_history[:, -self.sequence_length:, :].copy(), (1, self.sequence_length, self.input_dim))
            prediction.append(out[0,0,:]) # out has dimension [k,p,dim]

        print("\n")
        return prediction, target

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
        # SAVING RESULTS ON FILE
        return 0



