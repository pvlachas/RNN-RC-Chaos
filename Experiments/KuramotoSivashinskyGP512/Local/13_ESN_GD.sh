#!/bin/bash

cd ../../../Methods



python3 RUN.py esn_gd \
--mode all \
--display_output 1 \
--system_name KuramotoSivashinskyGP512 \
--write_to_log 0 \
--N 100000 \
--N_used 1000 \
--RDIM 3 \
--scaler Standard \
--degree 10 \
--radius 0.6 \
--sigma_input 1 \
--regularization 0.0 \
--dynamics_length 200 \
--iterative_prediction_length 200 \
--num_test_ICS 2 \
--number_of_epochs 100 \
--learning_rate 1e-8 \
--solver gd \
--approx_reservoir_size 50000 \
--noise_level 5

# solver : {"auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"}
