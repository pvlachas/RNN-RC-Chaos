#!/bin/bash

cd ../../../Methods



python3 RUN.py mlp \
--mode all \
--display_output 1 \
--system_name Lorenz96_F8GP40R40 \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--RDIM 40 \
--scaler Standard \
--initializer xavier \
--regularization 0.0 \
--sequence_length 1 \
--prediction_length 1 \
--mlp_activation_str tanh \
--mlp_num_layers 3 \
--mlp_size_layers 1000 \
--subsample 1 \
--batch_size 32 \
--max_epochs 1000 \
--num_rounds 10 \
--overfitting_patience 5 \
--training_min_epochs 1 \
--learning_rate 0.0001 \
--train_val_ratio 0.8 \
--iterative_prediction_length 100 \
--num_test_ICS 2 \
--batched_valtrain 0 \
--keep_prob 1 \
--retrain 0
