#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import socket

# Plotting parameters
import matplotlib
hostname = socket.gethostname()
print("PLOTTING HOSTNAME: {:}".format(hostname))
CLUSTER = True if ((hostname[:2]=='eu')  or (hostname[:5]=='daint') or (hostname[:3]=='nid')) else False
if CLUSTER: matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm
from mpl_toolkits import mplot3d

from matplotlib import colors
import six
color_dict = dict(six.iteritems(colors.cnames))

font = {'size'   : 16, 'family':'Times New Roman'}
matplotlib.rc('font', **font)


def plotTrainingLosses(model, loss_train, loss_val, min_val_error,additional_str=""):
    if (len(loss_train) != 0) and (len(loss_val) != 0):
        min_val_epoch = np.argmin(np.abs(np.array(loss_val)-min_val_error))
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/Loss_total"+ additional_str + ".png"
        fig, ax = plt.subplots()
        plt.title("Validation error {:.10f}".format(min_val_error))
        plt.plot(np.arange(np.shape(loss_train)[0]), loss_train, color=color_dict['green'], label="Train RMSE")
        plt.plot(np.arange(np.shape(loss_val)[0]), loss_val, color=color_dict['blue'], label="Validation RMSE")
        plt.plot(min_val_epoch, min_val_error, "o", color=color_dict['red'], label="optimal")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Loss")
        plt.legend()
        plt.savefig(fig_path)
        plt.close()

        fig_path = model.saving_path + model.fig_dir + model.model_name + "/Loss_total_log"+ additional_str + ".png"
        fig, ax = plt.subplots()
        plt.title("Validation error {:.10f}".format(min_val_error))
        plt.plot(np.arange(np.shape(loss_train)[0]), np.log(loss_train), color=color_dict['green'], label="Train RMSE")
        plt.plot(np.arange(np.shape(loss_val)[0]), np.log(loss_val), color=color_dict['blue'], label="Validation RMSE")
        plt.plot(min_val_epoch, np.log(min_val_error), "o", color=color_dict['red'], label="optimal")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Log-Loss")
        plt.legend()
        plt.savefig(fig_path)
        plt.close()

    else:
        print("## Empty losses. Not printing... ##")



def plotAttractor(model, set_name, latent_states, ic_idx):

    print(np.shape(latent_states))
    if np.shape(latent_states)[1] >= 2:
        fig, ax = plt.subplots()
        plt.title("Latent dynamics in {:}".format(set_name))
        X = latent_states[:, 0]
        Y = latent_states[:, 1]
        epsilon = 1e-7
        # for i in range(0, len(X)-1):
        for i in range(len(X)-1):
            if np.abs(X[i+1]-X[i]) > epsilon and np.abs(Y[i+1]-Y[i]) > epsilon:
                # plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='red', head_width=.05, shape='full', lw=0, length_includes_head=True, zorder=2, linestyle='')
                plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='red', head_width=.05, shape='full', length_includes_head=True, zorder=2)
                # plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='red', shape='full', zorder=2)
        plt.plot(X, Y, 'k', linewidth = 1, label='output', zorder=1)
        plt.autoscale(enable=True, axis='both')
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/lattent_dynamics_{:}_{:}.png".format(set_name, ic_idx)
        plt.savefig(fig_path, dpi=300)
        plt.close()
    else:
        fig, ax = plt.subplots()
        plt.title("Latent dynamics in {:}".format(set_name))
        plt.plot(latent_states[:-1, 0], latent_states[1:, 0], 'b', linewidth = 2.0, label='output')
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/lattent_dynamics_{:}_{:}.png".format(set_name, ic_idx)
        plt.savefig(fig_path, dpi=300)
        plt.close()



def plotIterativePrediction(model, set_name, target, prediction, error, nerror, ic_idx, dt, truth_augment=None, prediction_augment=None, warm_up=None, latent_states=None):


    if latent_states is not None:
        plotAttractor(model, set_name, latent_states, ic_idx)

    if ((truth_augment is not None) and (prediction_augment is not None)):
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_augmend_{:}_{:}.png".format(set_name, ic_idx)
        plt.plot(np.arange(np.shape(prediction_augment)[0]), prediction_augment[:,0], 'b', linewidth = 2.0, label='output')
        plt.plot(np.arange(np.shape(truth_augment)[0]), truth_augment[:,0], 'r', linewidth = 2.0, label='target')
        plt.plot(np.ones((100,1))*warm_up, np.linspace(np.min(truth_augment[:,0]), np.max(truth_augment[:,0]), 100), 'g--', linewidth = 2.0, label='warm-up')
        plt.legend(loc="lower right")
        plt.savefig(fig_path)
        plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}.png".format(set_name, ic_idx)
    plt.plot(prediction, 'r--', label='prediction')
    plt.plot(target, 'g--', label='target')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_error.png".format(set_name, ic_idx)
    plt.plot(error, label='error')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_log_error.png".format(set_name, ic_idx)
    plt.plot(np.log(np.arange(np.shape(error)[0])), np.log(error), label='log(error)')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_nerror.png".format(set_name, ic_idx)
    plt.plot(nerror, label='nerror')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()

    if model.input_dim >=3: createTestingContours(model, target, prediction, dt, ic_idx, set_name)


def createTestingContours(model, target, output, dt, ic_idx, set_name):
    fontsize = 12
    error = np.abs(target-output)
    # vmin = np.array([target.min(), output.min()]).min()
    # vmax = np.array([target.max(), output.max()]).max()
    vmin = target.min()
    vmax = target.max()
    vmin_error = 0.0
    vmax_error = target.max()

    print("VMIN: {:} \nVMAX: {:} \n".format(vmin, vmax))

    # Plotting the contour plot
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12, 6), sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace = 0.4)
    axes[0].set_ylabel(r"Time $t$", fontsize=fontsize)
    createContour_(fig, axes[0], target, "Target", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt)
    createContour_(fig, axes[1], output, "Output", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt)
    createContour_(fig, axes[2], error, "Error", fontsize, vmin_error, vmax_error, plt.get_cmap("Reds"), dt)
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_contour.png".format(set_name, ic_idx)
    plt.savefig(fig_path)
    plt.close()

def createContour_(fig, ax, data, title, fontsize, vmin, vmax, cmap, dt):
    ax.set_title(title, fontsize=fontsize)
    t, s = np.meshgrid(np.arange(data.shape[0])*dt, np.arange(data.shape[1]))
    mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both")
    fig.colorbar(mp, ax=ax)
    ax.set_xlabel(r"$State$", fontsize=fontsize)
    return mp

def plotSpectrum(model, sp_true, sp_pred, freq_true, freq_pred, set_name):
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/frequencies_{:}.png".format(set_name)
    plt.plot(freq_pred, sp_pred, 'r--', label="prediction")
    plt.plot(freq_true, sp_true, 'g--', label="target")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectrum [dB]')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()






