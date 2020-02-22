#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
import numpy as np

def Lorenz96(t,x0, F):
    dxdt = np.matrix(np.zeros(x0.shape))
    end = x0.shape[1]
    dxdt[:,2:end-1] = -np.multiply(x0[:,0:end-3],x0[:,1:end-2]) + np.multiply(x0[:,1:end-2],x0[:,3:end]) - x0[:,2:end-1]+F
    dxdt[:,0] = -np.multiply(x0[:,end-2],x0[:,end-1])+np.multiply(x0[:,end-1],x0[:,1])-x0[:,0]+F
    dxdt[:,1] = -np.multiply(x0[:,end-1],x0[:,0])+np.multiply(x0[:,0],x0[:,2])-x0[:,1]+F
    dxdt[:,end-1] = -np.multiply(x0[:,end-3],x0[:,end-2])+np.multiply(x0[:,end-2],x0[:,0])-x0[:,end-1]+F
    return dxdt

def RK4(dxdt, x0, t0, dt, F):
    k1 = dxdt(t0,x0, F);
    k2 = dxdt(t0+dt/2.,x0+dt*k1/2., F);
    k3 = dxdt(t0+dt/2.,x0+dt*k2/2., F);
    k4 = dxdt(t0+dt,x0+dt*k3, F);
    x = x0 + 1./6*(k1+2*k2+2*k3+k4)*dt;
    return x




