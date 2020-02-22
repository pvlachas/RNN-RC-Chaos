#!/bin/bash

cd ../../../Methods

python3 RUN.py rnn_statefull \
--mode all \
--display_output 0 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 8000 \
--RDIM 3 \
--rnn_cell_type gru \
--unitary_cplex 1 \
--unitary_capacity 2 \
--regularization 0.0 \
--scaler standard \
--initializer xavier \
--sequence_length 4 \
--hidden_state_propagation_length 1000 \
--prediction_length 1 \
--rnn_activation_str tanh \
--rnn_num_layers 2 \
--rnn_size_layers 64 \
--dropout_keep_prob 0.999 \
--zoneout_keep_prob 0.999 \
--subsample 1 \
--batch_size 32 \
--max_epochs 100 \
--num_rounds 3 \
--overfitting_patience 30 \
--training_min_epochs 1 \
--learning_rate 0.01 \
--train_val_ratio 0.8 \
--iterative_prediction_length 200 \
--num_test_ICS 1 \
--retrain 0 \
--noise_level 10
