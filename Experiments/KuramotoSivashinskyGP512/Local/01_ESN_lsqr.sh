#!/bin/bash

cd ../../../Methods



python3 RUN.py esn \
--mode all \
--display_output 1 \
--system_name KuramotoSivashinskyGP512 \
--write_to_log 0 \
--N 100000 \
--N_used 1000 \
--RDIM 3 \
--scaler Standard \
--approx_reservoir_size 1000 \
--degree 10 \
--radius 0.6 \
--sigma_input 1 \
--regularization 0.0 \
--dynamics_length 200 \
--iterative_prediction_length 200 \
--noise_level 0 \
--num_test_ICS 2 \
--solver lsqr \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5


# for RDIM in 1
# do
# for ARS in 2000
# do
# for SI in 0.5 1.0 1.5
# do
# python3 RUN.py esn \
# --mode all \
# --display_output 1 \
# --system_name Lorenz3D \
# --write_to_log 1 \
# --N 100000 \
# --N_used 1000 \
# --RDIM $RDIM \
--noise_level 1 \
# --scaler Standard \
# --approx_reservoir_size $ARS \
# --degree 8 \
# --radius 0.95 \
# --sigma_input $SI \
# --regularization 0.0 \
# --dynamics_length 200 \
# --iterative_prediction_length 100 \
# --num_test_ICS 10
# done
# done
# done

