#!/bin/bash

cd ../../../Methods


python3 RUN.py md_arnn \
--mode all \
--system_name KuramotoSivashinskyGP512 \
--write_to_log 0 \
--N 100000 \
--N_used 100000 \
--input_dim 3 \
--skip 5 \
--md 0 \
--dynamics_loss 0 \
--train_val_ratio 0.8 \
--rnn_cell_type lstm \
--rnn_layers 32  \
--rnn_activation_str tanh \
--autoencoder_layers 50 50 50 \
--autoencoder_activation_str selu \
--latent_state_dim 2  \
--latent_noise_dim 0  \
--zoneout_keep_prob 1 \
--sequence_length 32 \
--hidden_state_propagation_length 600 \
--scaler Standard \
--noise_level 5 \
--learning_rate 0.001 \
--batch_size 16 \
--overfitting_patience 20 \
--training_min_epochs 1 \
--max_epochs 100 \
--num_rounds 3 \
--num_test_ICS 2 \
--iterative_prediction_length 80 \
--display_output 1 \
--random_seed 12 \
--trackLatentState 1 \
--retrain 1

# TODO: SKIP, DATA FROM MANY ICS, ZONE-OUT



