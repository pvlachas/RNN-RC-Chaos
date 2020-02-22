#!/bin/bash

cd ../../../Methods

for RDIM in 40
do
for ARS in 3000
do
for SI in 0.5
do
python3 RUN.py esn \
--mode all \
--display_output 1 \
--system_name Lorenz96_F10GP40R40 \
--write_to_log 1 \
--N 100000 \
--N_used 10000 \
--RDIM $RDIM \
--noise_level 10 \
--scaler Standard \
--approx_reservoir_size $ARS \
--degree 10 \
--radius 0.6 \
--sigma_input $SI \
--regularization 0.0 \
--dynamics_length 1000 \
--iterative_prediction_length 100 \
--num_test_ICS 2
done
done
done
