#!/bin/bash

cd ../../../Methods

for RDIM in 40
do
for SS in 60
do
for SL in 8
do
python3 RUN.py rnn_statefull \
--mode all \
--display_output 1 \
--system_name Lorenz96_F8GP40R40 \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--RDIM $RDIM \
--noise_level 1 \
--rnn_cell_type unitary \
--unitary_cplex 1 \
--unitary_capacity 2 \
--regularization 0.0 \
--keep_prob 1.0 \
--scaler standard \
--initializer xavier \
--sequence_length $SL \
--hidden_state_propagation_length 300 \
--prediction_length 1 \
--rnn_activation_str tanh \
--rnn_num_layers 1 \
--rnn_size_layers $SS \
--subsample 1 \
--batch_size 32 \
--max_epochs 20 \
--num_rounds 5 \
--overfitting_patience 5 \
--training_min_epochs 1 \
--learning_rate 0.1 \
--train_val_ratio 0.8 \
--iterative_prediction_length 100 \
--num_test_ICS 2 \
--retrain 0
done
done
done


# python3 RUN.py rnn_statefull \
# --mode all \
# --display_output 0 \
# --system_name Lorenz3D \
# --write_to_log 0 \
# --N 100000 \
# --N_used 5000 \
# 
# --rnn_cell_type unitary \
# --unitary_cplex 1 \
# --unitary_capacity 2 \
# --regularization 0.0 \
# --scaler standard \
# --initializer xavier \
# --sequence_length 12 \
# --hidden_state_propagation_length 100 \
# --prediction_length 8 \
# --rnn_num_layers 1 \
# --rnn_size_layers 128 \
# --subsample 1 \
# --batch_size 32 \
# --max_epochs 20 \
# --num_rounds 20 \
# --overfitting_patience 2 \
# --training_min_epochs 2 \
# --learning_rate 0.001 \
# --train_val_ratio 0.8 \
# --iterative_prediction_length 400 \
# --num_test_ICS 3 \
# --retrain 0
