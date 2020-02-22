#!/bin/bash


# module load daint-gpu
# module load slurm
# module load cray-mpich
# module load TensorFlow/1.7.0-CrayGNU-18.08-cuda-9.1-python3
# source ~/venv-python3.6/bin/activate


cd ../../../Methods



for RDIM in 40
do
for SS in 1000
do
for SL in 8
do
python3 RUN.py rnn_statefull \
--mode all \
--display_output 1 \
--system_name Lorenz96_F8GP40R40 \
--write_to_log 1 \
--N 100000 \
--N_used 40000 \
--RDIM $RDIM \
--noise_level 1 \
--rnn_cell_type gru \
--unitary_cplex 1 \
--unitary_capacity 2 \
--regularization 0.0 \
--keep_prob 1.0 \
--scaler standard \
--initializer xavier \
--sequence_length $SL \
--hidden_state_propagation_length 1250 \
--prediction_length 1 \
--rnn_activation_str tanh \
--rnn_num_layers 1 \
--rnn_size_layers $SS \
--subsample 1 \
--batch_size 32 \
--max_epochs 100 \
--num_rounds 16 \
--overfitting_patience 5 \
--training_min_epochs 1 \
--learning_rate 0.001 \
--train_val_ratio 0.8 \
--iterative_prediction_length 100 \
--num_test_ICS 2 \
--retrain 0
done
done
done

# for RDIM in 40; do for SS in 100; do for SL in 8; do python3 RUN.py rnn_statefull --mode test --display_output 1 --system_name Lorenz96_F8GP40R40 --write_to_log 1 --N 100000 --N_used 10000 --RDIM $RDIM --rnn_cell_type gru --unitary_cplex 1 --unitary_capacity 2 --regularization 0.0 --scaler standard --initializer xavier --sequence_length $SL --hidden_state_propagation_length 1200 --prediction_length $SL --rnn_activation_str tanh --rnn_num_layers 1 --rnn_size_layers $SS --subsample 1 --batch_size 32 --max_epochs 100 --num_rounds 16 --overfitting_patience 5 --training_min_epochs 1 --learning_rate 10 --train_val_ratio 0.8 --iterative_prediction_length 100 --num_test_ICS 10 --retrain 0; done; done; done


# python3 RUN.py rnn_statefull \
# --mode all \
# --display_output 0 \
# --system_name Lorenz3D \
# --write_to_log 1 \
# --N 100000 \
# --N_used 5000 \
# 
# --rnn_cell_type gru \
# --unitary_cplex 1 \
# --unitary_capacity 2 \
# --regularization 0.0 \
# --scaler standard \
# --initializer xavier \
# --sequence_length 12 \
# --hidden_state_propagation_length 800 \
# --prediction_length 6 \
# --rnn_activation_str tanh \
# --rnn_num_layers 1 \
# --rnn_size_layers 64 \
# --subsample 1 \
# --batch_size 32 \
# --max_epochs 10 \
# --num_rounds 5 \
# --overfitting_patience 10 \
# --training_min_epochs 10 \
# --learning_rate 0.01 \
# --train_val_ratio 0.9 \
# --iterative_prediction_length 1000 \
# --num_test_ICS 3 \
# --retrain 0


