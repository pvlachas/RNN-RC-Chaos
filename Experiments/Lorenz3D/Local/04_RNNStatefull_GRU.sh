#!/bin/bash


cd ../../../Methods

for RDIM in 3
do
for SS in 30
do
for SL in 1
do
for KP in 1.0
do
python3 RUN.py rnn_statefull \
--mode all \
--display_output 1 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--RDIM $RDIM \
--noise_level 1 \
--rnn_cell_type gru \
--unitary_cplex 1 \
--unitary_capacity 2 \
--regularization 0.0 \
--scaler standard \
--initializer xavier \
--sequence_length $SL \
--hidden_state_propagation_length 300 \
--prediction_length $SL \
--rnn_activation_str tanh \
--rnn_num_layers 2 \
--rnn_size_layers $SS \
--dropout_keep_prob $KP \
--zoneout_keep_prob $KP \
--subsample 1 \
--batch_size 32 \
--max_epochs 2 \
--num_rounds 1 \
--overfitting_patience 20 \
--training_min_epochs 1 \
--learning_rate 0.001 \
--train_val_ratio 0.8 \
--iterative_prediction_length 100 \
--num_test_ICS 2 \
--retrain 0
done
done
done
done

