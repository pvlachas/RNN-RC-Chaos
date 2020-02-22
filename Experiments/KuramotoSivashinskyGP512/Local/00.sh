#!/bin/bash

cd ../../../Methods

for RDIM in 1
do
for SS in 3
do
for SL in 32
do
for KP in 1.0
do
python3 RUN.py rnn_statefull \
--mode all \
--display_output 1 \
--system_name KuramotoSivashinskyGP512 \
--write_to_log 1 \
--N 100000 \
--N_used 100000 \
--RDIM $RDIM \
--noise_level 1 \
--rnn_cell_type lstm \
--unitary_cplex 1 \
--unitary_capacity 2 \
--regularization 0.0 \
--scaler standard \
--initializer xavier \
--sequence_length $SL \
--dropout_keep_prob $KP \
--zoneout_keep_prob $KP \
--hidden_state_propagation_length 1000 \
--prediction_length $SL \
--rnn_activation_str tanh \
--rnn_num_layers 1 \
--rnn_size_layers $SS \
--subsample 5 \
--batch_size 32 \
--max_epochs 1000 \
--num_rounds 5 \
--overfitting_patience 20 \
--training_min_epochs 1 \
--learning_rate 0.1 \
--train_val_ratio 0.75 \
--iterative_prediction_length 2000 \
--num_test_ICS 2 \
--retrain 0 \
--trackHiddenState 1
done
done
done
done



