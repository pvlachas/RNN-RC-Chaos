#!/bin/bash

cd ../../../Methods

mpiexec -n 3 python3 RUN.py rnn_statefull_parallel \
--mode all \
--display_output 0 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--RDIM 3 \
--rnn_cell_type unitary \
--unitary_cplex 1 \
--unitary_capacity 2 \
--regularization 0.0 \
--scaler standard \
--initializer xavier \
--sequence_length 12 \
--hidden_state_propagation_length 300 \
--dropout_keep_prob 0.995 \
--zoneout_keep_prob 0.995 \
--prediction_length 6 \
--rnn_num_layers 1 \
--rnn_size_layers 1000 \
--subsample 1 \
--batch_size 32 \
--max_epochs 100 \
--num_rounds 5 \
--overfitting_patience 100 \
--training_min_epochs 1 \
--learning_rate 0.001 \
--train_val_ratio 0.8 \
--num_parallel_groups 3 \
--parallel_group_interaction_length 1 \
--iterative_prediction_length 200 \
--num_test_ICS 1 \
--reference_train_time 1 \
--buffer_train_time 0 \
--retrain 0 \
--noise_level 5
