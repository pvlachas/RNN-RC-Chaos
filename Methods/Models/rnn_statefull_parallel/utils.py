#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import sklearn.decomposition as decomposition
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
import random as rand
import pickle
import argparse
import os
from sklearn.utils import shuffle
import sklearn.preprocessing as sk

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest


def best_effort_input_batch_size(flat_input):
    for input_ in flat_input:
        shape = input_.shape
        if shape.ndims is None:
            continue
        if shape.ndims < 2:
            raise ValueError("Expected input tensor %s to have rank at least 2" % input_)
        batch_size = shape[1].value
        if batch_size is not None:
            return batch_size
            # Fallback to the dynamic batch size of the first input.
        return array_ops.shape(flat_input[0])[1]

def transpose_batch_time(x):
  """Transposes the batch and time dimensions of a Tensor.
  If the input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.
  Args:
    x: A Tensor.
  Returns:
    x transposed along the first two dimensions.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    return x

  x_rank = array_ops.rank(x)
  x_t = array_ops.transpose(
      x, array_ops.concat(
          ([1, 0], math_ops.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tensor_shape.TensorShape([
          x_static_shape[1].value, x_static_shape[0].value
      ]).concatenate(x_static_shape[2:]))
  return x_t

