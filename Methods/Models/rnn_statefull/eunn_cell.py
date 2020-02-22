from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
from tensorflow.python.ops import rnn_cell_impl


def modrelu(inputs, bias, cplex=True, dtypec=tf.complex128):
    """
    modReLU activation function
    """
    if cplex:
        norm = tf.abs(inputs) + 0.00001
        biased_norm = norm + bias
        magnitude = tf.cast(tf.nn.relu(biased_norm), dtype=dtypec)
        phase = inputs / tf.cast(norm, dtype=dtypec)
    else:
        norm = tf.abs(inputs) + 0.00001
        biased_norm = norm + bias
        magnitude = tf.nn.relu(biased_norm)
        phase = tf.sign(inputs)
       
    return phase * magnitude

def generate_index_tunable(s, L):
    """
    generate the index lists for eunn to prepare weight matrices 
    and perform efficient rotations
    This function works for tunable case
    """    
    ind1 = list(range(s))
    ind2 = list(range(s))

    for i in range(s):
        if i%2 == 1:
            ind1[i] = ind1[i] - 1
            if i == s -1:
                continue
            else:
                ind2[i] = ind2[i] + 1
        else:
            ind1[i] = ind1[i] + 1
            if i == 0: 
                continue
            else:
                ind2[i] = ind2[i] - 1

    ind_exe = [ind1, ind2] * int(L/2)

    ind3 = []
    ind4 = []

    for i in range(int(s/2)):
        ind3.append(i)
        ind3.append(i + int(s/2))

    ind4.append(0)
    for i in range(int(s/2) - 1):
        ind4.append(i + 1)
        ind4.append(i + int(s/2))
    ind4.append(s - 1)

    ind_param = [ind3, ind4]

    return ind_exe, ind_param


def generate_index_fft(s):
    """
    generate the index lists for eunn to prepare weight matrices 
    and perform efficient rotations
    This function works for fft case
    """      
    def ind_s(k):
        if k==0:
            return np.array([[1,0]])
        else:
            temp = np.array(range(2**k))
            list0 = [np.append(temp + 2**k, temp)]
            list1 = ind_s(k-1)
            for i in range(k):
                list0.append(np.append(list1[i],list1[i] + 2**k))
            return list0

    t = ind_s(int(math.log(s/2, 2)))

    ind_exe = []
    for i in range(int(math.log(s, 2))):
        ind_exe.append(tf.constant(t[i]))

    ind_param = []
    for i in range(int(math.log(s, 2))):
        ind = np.array([])
        for j in range(2**i):
            ind = np.append(ind, np.array(range(0, s, 2**i)) + j).astype(np.int32)

        ind_param.append(tf.constant(ind))
    
    return ind_exe, ind_param


def fft_param(model, num_units, cplex, dtype, dtypec):
    
    phase_init = tf.random_uniform_initializer(-3.14, 3.14)
    capacity = int(math.log(num_units, 2))

    theta = tf.get_variable("theta_layer_{:d}".format(model._ln), [capacity, num_units//2], 
        initializer=phase_init, dtype=dtype)
    cos_theta = tf.cos(theta)
    sin_theta = tf.sin(theta)
        
    if cplex:
        phi = tf.get_variable("phi_layer_{:d}".format(model._ln), [capacity, num_units//2], 
            initializer=phase_init, dtype=dtype)
        cos_phi = tf.cos(phi)
        sin_phi = tf.sin(phi)

        cos_list_re = tf.concat([cos_theta, cos_theta * cos_phi], axis=1)
        cos_list_im = tf.concat([tf.zeros_like(theta), cos_theta * sin_phi], axis=1)
        sin_list_re = tf.concat([sin_theta, - sin_theta * cos_phi], axis=1)
        sin_list_im = tf.concat([tf.zeros_like(theta), - sin_theta * sin_phi], axis=1)
        cos_list = tf.complex(cos_list_re, cos_list_im)
        sin_list = tf.complex(sin_list_re, sin_list_im)

    else:
        cos_list = tf.concat([cos_theta, cos_theta], axis=1)
        sin_list = tf.concat([sin_theta, -sin_theta], axis=1)

        
    ind_exe, index_fft = generate_index_fft(num_units)

    v1 = tf.stack([tf.gather(cos_list[i,:], index_fft[i]) for i in range(capacity)])
    v2 = tf.stack([tf.gather(sin_list[i,:], index_fft[i]) for i in range(capacity)])

    if cplex:
        omega = tf.get_variable("omega_layer_{:d}".format(model._ln), [num_units], 
                initializer=phase_init, dtype=dtype)
        D = tf.complex(tf.cos(omega), tf.sin(omega))
    else:
        D = None

    diag = D

    return v1, v2, ind_exe, diag

def tunable_param(model, num_units, cplex, capacity, dtype, dtypec):

    capacity_A = int(capacity//2)
    capacity_B = capacity - capacity_A
    phase_init = tf.random_uniform_initializer(-3.14, 3.14, dtype=dtype)

    theta_A = tf.get_variable("theta_A_layer_{:d}".format(model._ln), [capacity_A, num_units//2], 
        initializer=phase_init, dtype=dtype)
    cos_theta_A = tf.cos(theta_A)
    sin_theta_A = tf.sin(theta_A)

    if cplex:
        phi_A = tf.get_variable("phi_A_layer_{:d}".format(model._ln), [capacity_A, num_units//2], 
            initializer=phase_init, dtype=dtype)
        cos_phi_A = tf.cos(phi_A)
        sin_phi_A = tf.sin(phi_A)

        cos_list_A_re = tf.concat([cos_theta_A, cos_theta_A * cos_phi_A], axis=1)
        cos_list_A_im = tf.concat([tf.zeros_like(theta_A), cos_theta_A * sin_phi_A], axis=1)
        sin_list_A_re = tf.concat([sin_theta_A, - sin_theta_A * cos_phi_A], axis=1)
        sin_list_A_im = tf.concat([tf.zeros_like(theta_A), - sin_theta_A * sin_phi_A], axis=1)
        cos_list_A = tf.complex(cos_list_A_re, cos_list_A_im)
        sin_list_A = tf.complex(sin_list_A_re, sin_list_A_im)
    else:
        cos_list_A = tf.concat([cos_theta_A, cos_theta_A], axis=1)
        sin_list_A = tf.concat([sin_theta_A, -sin_theta_A], axis=1)         

    theta_B = tf.get_variable("theta_B_layer_{:d}".format(model._ln), [capacity_B, num_units//2 - 1], 
        initializer=phase_init, dtype=dtype)
    cos_theta_B = tf.cos(theta_B)
    sin_theta_B = tf.sin(theta_B)

    if cplex:
        phi_B = tf.get_variable("phi_B_layer_{:d}".format(model._ln), [capacity_B, num_units//2 - 1], 
            initializer=phase_init, dtype=dtype)
        cos_phi_B = tf.cos(phi_B)
        sin_phi_B = tf.sin(phi_B)

        cos_list_B_re = tf.concat([tf.ones([capacity_B, 1], dtype=dtype), 
            cos_theta_B, cos_theta_B * cos_phi_B, tf.ones([capacity_B, 1], dtype=dtype)], axis=1)
        cos_list_B_im = tf.concat([tf.zeros([capacity_B, 1], dtype=dtype), tf.zeros_like(theta_B), 
            cos_theta_B * sin_phi_B, tf.zeros([capacity_B, 1], dtype=dtype)], axis=1)
        sin_list_B_re = tf.concat([tf.zeros([capacity_B, 1], dtype=dtype), sin_theta_B, 
            - sin_theta_B * cos_phi_B, tf.zeros([capacity_B, 1], dtype=dtype)], axis=1)
        sin_list_B_im = tf.concat([tf.zeros([capacity_B, 1], dtype=dtype), tf.zeros_like(theta_B), 
            - sin_theta_B * sin_phi_B, tf.zeros([capacity_B, 1], dtype=dtype)], axis=1)
        cos_list_B = tf.complex(cos_list_B_re, cos_list_B_im)
        sin_list_B = tf.complex(sin_list_B_re, sin_list_B_im)
    else:
        cos_list_B = tf.concat([tf.ones([capacity_B, 1], dtype=dtype), cos_theta_B, 
            cos_theta_B, tf.ones([capacity_B, 1], dtype=dtype)], axis=1)
        sin_list_B = tf.concat([tf.zeros([capacity_B, 1], dtype=dtype), sin_theta_B, 
            - sin_theta_B, tf.zeros([capacity_B, 1], dtype=dtype)], axis=1)


    ind_exe, [index_A, index_B] = generate_index_tunable(num_units, capacity)
 

    diag_list_A = tf.gather(cos_list_A, index_A, axis=1)
    off_list_A = tf.gather(sin_list_A, index_A, axis=1)
    diag_list_B = tf.gather(cos_list_B, index_B, axis=1)
    off_list_B = tf.gather(sin_list_B, index_B, axis=1)


    v1 = tf.reshape(tf.concat([diag_list_A, diag_list_B], axis=1), [capacity, num_units])
    v2 = tf.reshape(tf.concat([off_list_A, off_list_B], axis=1), [capacity, num_units])


    if cplex:
        omega = tf.get_variable("omega_layer_{:d}".format(model._ln), [num_units], 
            initializer=phase_init, dtype=dtype)
        D = tf.complex(tf.cos(omega), tf.sin(omega))
    else:
        D = None

    diag = D

    return v1, v2, ind_exe, diag




class EUNNCell(rnn_cell_impl.RNNCell):
    """Efficient Unitary Network Cell
    
    The implementation is based on: 

    http://arxiv.org/abs/1612.05231.

    """
    def __init__(self, 
                num_units, 
                capacity=2, 
                fft=False, 
                cplex=True,
                dtype=tf.float64,
                reuse=None,
                ln=0):
        """Initializes the EUNN  cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          capacity: int, The capacity of the unitary matrix for tunable
            case.
          fft: bool, default false, whether to use fft style 
          architecture or tunable style.
          cplex: bool, default true, whether to use cplex number.
        """

        super(EUNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._capacity = capacity
        self._fft = fft
        self._cplex = cplex
        self._dtype = dtype
        self._dtypec = tf.complex128 if self._dtype == tf.float64 else tf.complex64
        self._ln = ln

        if self._capacity > self._num_units:
            raise ValueError("Do not set capacity larger than hidden size, it is redundant")

        if self._fft:
            if math.log(self._num_units, 2) % 1 != 0: 
                raise ValueError("FFT style only supports power of 2 of hidden size")
        else:
            if self._num_units % 2 != 0:
                raise ValueError("Tunable style only supports even number of hidden size")

            if self._capacity % 2 != 0:
                raise ValueError("Tunable style only supports even number of capacity")



        if self._fft:
            self._capacity = int(math.log(self._num_units, 2))
            self._v1, self._v2, self._ind, self._diag = fft_param(self, self._num_units, self._cplex, self._dtype, self.dtypec)
        else:
            self._v1, self._v2, self._ind, self._diag = tunable_param(self, self._num_units, self._cplex, self._capacity, self._dtype, self._dtypec)


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


    def loop(self, h):
        for i in range(self._capacity):
            diag = h * self._v1[i,:]
            off = h * self._v2[i,:]
            h = diag + tf.gather(off, self._ind[i], axis=1)

        if self._diag is not None:
            h = h * self._diag

        return h


    def __call__(self, inputs, state, scope=None, reuse=None):
        with tf.variable_scope(scope or "eunn_cell", reuse=reuse):

            inputs_size = inputs.get_shape()[-1]

            # state = _eunn_loop(state, self._capacity, self.diag_vec, self.off_vec, self.diag, self._fft)

            # inputs to hidden
            input_matrix_init = tf.random_uniform_initializer(-0.01, 0.01)
            if self._cplex:
                U_re = tf.get_variable("U_re_layer_{:d}".format(self._ln), [inputs_size, self._num_units], initializer=input_matrix_init, dtype=self._dtype)
                U_im = tf.get_variable("U_im_layer_{:d}".format(self._ln), [inputs_size, self._num_units], initializer=input_matrix_init, dtype=self._dtype)
                inputs_re = tf.matmul(inputs, U_re)
                inputs_im = tf.matmul(inputs, U_im)
                inputs = tf.complex(inputs_re, inputs_im)
            else:
                U = tf.get_variable("U_layer_{:d}".format(self._ln), [inputs_size, self._num_units], 
                    initializer=input_matrix_init, dtype=self._dtype)
                inputs = tf.matmul(inputs, U)

            # hidden to hidden
            state = self.loop(state)

            # activation
            bias = tf.get_variable("mod_ReLU_bias_layer_{:d}".format(self._ln), [self._num_units], 
                    initializer=tf.constant_initializer(), dtype=self._dtype)
            output = modrelu((inputs + state), bias, self._cplex, self._dtypec)

        return output, output
