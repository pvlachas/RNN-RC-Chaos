#!/bin/bash

cd ../../../Methods

for input_dim in 3
do
# for SS in 100
# do
# for KP in 0.999 0.995 0.99 0.98
# do
python3 RUN.py deep_esn \
--mode all \
--display_output 0 \
--system_name KuramotoSivashinskyGP512 \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--input_dim $input_dim \
--regularization 0.1 \
--scaler standard \
--noise_level 5 \
--num_test_ICS 2 \
--radius .9999 \
--sigma 2 \
--dynamics_length 200 \
--iterative_prediction_length 1000 \
--leaky_rate .995 \
--input_scale 1 \
--Nr 50 \
--Nl 20 \
--verbose 1 \
--pretrain 0 \
--pretrain_threshold .1 \
--pretrain_Nepochs 50 \
--pretrain_learning_rate 1e-4 \
--reservoir_connectivity 1 \
--readout_trainMethod "ridge" \
--readout_solver "lsqr" \
--transient 0 \
# --degree .1 \
# --hidden_state_propagation_length 300 \
# --prediction_length $PL \
# --subsample 1 \
# --batch_size 32 \
# --max_epochs 5 \
# --num_rounds 10 \
# --overfitting_patience 20 \
# --training_min_epochs 1 \
# --learning_rate 0.001 \
# --train_val_ratio 0.8 \
# --retrain 0 \
# --num_hidden_units 1000 \
# --sequence_length $SL \
# --initializer xavier \
# --rnn_size_layers $SS \
# --rnn_num_layers 2 \
# --dropout_keep_prob $KP \
# --zoneout_keep_prob $KP \
# --rnn_activation_str tanh \
# --rnn_cell_type gru \
# --unitary_cplex 1 \
# --unitary_capacity 2 \
# done
# done
done

