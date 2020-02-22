#!/bin/bash

cd ../../../Methods

python3 RUN.py esn_sgd \
--mode all \
--display_output 0 \
--system_name KuramotoSivashinskyGP512 \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--input_dim 3 \
--scaler standard \
--sequence_length 2 \
--hidden_state_propagation_length 600 \
--prediction_length 1 \
--subsample 1 \
--batch_size 16 \
--min_epochs 0 \
--overfitting_patience 20 \
--max_epochs 500 \
--num_rounds 4 \
--initial_learning_rate 0.001 \
--train_val_ratio 0.8 \
--iterative_prediction_length 300 \
--noise_level 0 \
--retrain 0 \
--num_hidden_units 1000 \
--radius .6 \
--degree 10 \
--sigma 0.1 \
--dropout 0 \
--num_test_ICS 3 \
