#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
import numpy as np

def lorenz(t0, u0, sigma, rho, beta):
    dudt = np.zeros(np.shape(u0))
    dudt[0] = sigma * (u0[1]-u0[0])
    dudt[1] = u0[0] * (rho-u0[2]) - u0[1]
    dudt[2] = u0[0] * u0[1] - beta*u0[2]
    return dudt













