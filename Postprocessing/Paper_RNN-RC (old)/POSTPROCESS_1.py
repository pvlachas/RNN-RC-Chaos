#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import pickle
import glob, os
import numpy as np
import argparse

# ADDING PARENT DIRECTORY TO PATH
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
methods_dir = os.path.dirname(current_dir)+"/Methods"
sys.path.insert(0, methods_dir) 
from Config.global_conf import global_params
global_utils_path = methods_dir + "/Models/Utils"
sys.path.insert(0, global_utils_path) 
from global_utils import *

# PLOTTING
import matplotlib
matplotlib.use('Agg')

font = {'size'   : 16, 'family':'Times New Roman'}
matplotlib.rc('font', **font)
linewidth = 3
markersize = 10
# # matplotlib.rc('text', usetex=True)

import matplotlib.pyplot as plt
plt.switch_backend('Agg')


import re
from Utils.utils import *



from scipy.spatial import ConvexHull


# PLOTTING THE NUM OF ACCURATE PREDICTIONS W.R.T. NUM MODEL PARAMETERS

# python3 POSTPROCESS_1.py --system_name Lorenz3D
# python3 POSTPROCESS_1.py --system_name Lorenz96_F8GP40R40 --Experiment_Name="Experiment_Daint_Large"


parser = argparse.ArgumentParser()
parser.add_argument("--system_name", help="system", type=str, required=True)
parser.add_argument("--Experiment_Name", help="Experiment_Name", type=str, required=False, default=None)
args = parser.parse_args()
system_name = args.system_name
Experiment_Name = args.Experiment_Name

# system_name="Lorenz3D"
# Experiment_Name="Experiment_Daint_Large"


if Experiment_Name is None or Experiment_Name=="None" or global_params.cluster == "daint":
    saving_path = global_params.saving_path.format(system_name)
else:
    saving_path = global_params.saving_path.format(Experiment_Name +"/"+system_name)
logfile_path=saving_path+"/Logfiles"
print(system_name)
print(logfile_path)

fig_path = saving_path + "/Total_Results_Figures"
os.makedirs(fig_path, exist_ok=True)

model_test_dict = getAllModelsTestDict(logfile_path)
model_train_dict = getAllModelsTrainDict(logfile_path)

# print(model_test_dict)
# print(ark)
if system_name == "Lorenz3D":
    RDIM_VALUES = [1,2,3]
    xlims = [0, 20000]
elif system_name == "Lorenz96_F8GP40R40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [35,40]
    xlims = None
elif system_name == "Lorenz96_F8GP40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [40]
    xlims = None
elif system_name == "Lorenz96_F10GP40R40":
    dt=0.01
    lambda1=2.27
    RDIM_VALUES = [35,40]
    xlims = None

# GPU-RNN-esn_auto-RDIM_35-N_used_100000-SIZE_6000-D_8.0-RADIUS_0.4-SIGMA_0.5-DL_2000-NL_0-IPL_2000-REG_0.0001-WID_0
# GPU-RNN-gru-RDIM_35-N_used_100000-NUM-LAY_1-SIZE-LAY_1000-ACT_tanh-ISH_statefull-SL_16-PL_1-LR_0.01-DKP_1.0-ZKP_1.0-HSPL_3125-IPL_2000-NL_2-WID_0
for NUM_ACC_PRED_STR in ["num_accurate_pred_050_avg_TEST"]:
    for RDIM in RDIM_VALUES:
    # for RDIM in [RDIM_VALUES[0]]:

        model_names_str = [
        "GPU-RNN-esn(.*)-RDIM_"+str(RDIM)+"-N_used_(.*)-SIZE_(.*)-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)(.*)-IPL_(.*)-REG_(.*)-WID_0",
        "GPU-RNN-gru-RDIM_"+str(RDIM)+"-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
        "GPU-RNN-lstm-RDIM_"+str(RDIM)+"-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
        "GPU-RNN-unitary-RDIM_"+str(RDIM)+"-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
        ]

        model_color_str = [
        "blue", 
        "green",
        "red",
        "orange",
        "blueviolet",
        "black",
        "cornflowerblue",
        ]
        model_marker_str = [
        "s", 
        "x", 
        "o",
        "d",
        "*", 
        "<",
        ">",
        ]
        model_marker_width_str = [
        2, 
        linewidth, 
        2,
        2,
        2, 
        2,
        2,
        ]
        model_label_str = [
        "RC", 
        "GRU",
        "LSTM",
        "Unit",
        ]
        plot_every_str = [
        1, 
        1,
        1,
        1,
        1,
        1,
        1,
        ]

        # plot_every_str = [
        # 50, 
        # 20,
        # 20,
        # 20,
        # 20,
        # 20,
        # 20,
        # ]

        N = len(model_names_str)
        # N = 2

        valid_time_vecs = {}
        n_TRAINPARAMS_vecs = {}
        n_MODELPARAMS_vecs = {}
        for MODEL in range(N):
            model_name = model_names_str[MODEL]
            model_color = model_color_str[MODEL]
            model_marker = model_marker_str[MODEL]
            model_markeredgewidth = model_marker_width_str[MODEL]
            model_label = model_label_str[MODEL]
            plot_every = plot_every_str[MODEL]

            valid_time_vec = []
            n_TRAINPARAMS_vec = []
            n_MODELPARAMS_vec = []
            # print(model_name)
            REGEX = re.compile(model_name)
            for model_name_key in model_test_dict:
                # print(model_name_key)
                if model_name_key in model_train_dict:
                    # print("MODEL FOUND!")
                    model_train = model_train_dict[model_name_key]
                    # print(model_train)
                    # print("################")
                    # print(REGEX)
                    # print(model_name_key)
                    # print(REGEX.match(model_name_key))
                    # print("################")
                    if REGEX.match(model_name_key):
                        # print("MATCHED")
                        # print(model_test_dict[model_name_key])
                        valid_time = model_test_dict[model_name_key][NUM_ACC_PRED_STR]*dt*lambda1
                        valid_time_vec.append(valid_time)
                        # print(valid_time)
                        n_trainable_params = model_train_dict[model_name_key]["n_trainable_parameters"]
                        # print(n_trainable_params)
                        n_model_parameters = model_train_dict[model_name_key]["n_model_parameters"]
                        n_TRAINPARAMS_vec.append(n_trainable_params)
                        # print(n_model_parameters)
                        # print(ark)
                        n_MODELPARAMS_vec.append(n_model_parameters)

                else:
                    print("MODEL NOT FOUND!")

            valid_time_vecs[model_name] = valid_time_vec
            n_TRAINPARAMS_vecs[model_name] = n_TRAINPARAMS_vec
            n_MODELPARAMS_vecs[model_name] = n_MODELPARAMS_vec

        # print(n_MODELPARAMS_vecs)
        # print(ark)

        fig, ax = plt.subplots(1)
        for MODEL in range(N):
            model_name = model_names_str[MODEL]
            model_color = model_color_str[MODEL]
            model_marker = model_marker_str[MODEL]
            model_markeredgewidth = model_marker_width_str[MODEL]
            model_label = model_label_str[MODEL]
            valid_time_vec = valid_time_vecs[model_name]
            n_TRAINPARAMS_vec = n_TRAINPARAMS_vecs[model_name]
            n_MODELPARAMS_vec = n_MODELPARAMS_vecs[model_name]
            # for i in range(np.shape(n_MODELPARAMS_vec)[0]):
                # ax.plot(n_TRAINPARAMS_vec[i], valid_time_vec[i], color=model_color, marker=model_marker, markersize=markersize, linewidth=linewidth, markeredgewidth=model_markeredgewidth)
            n_TRAINPARAMS_vec = np.reshape(n_TRAINPARAMS_vec, (-1,1))
            valid_time_vec = np.reshape(valid_time_vec, (-1,1))
            data = np.concatenate((n_TRAINPARAMS_vec, valid_time_vec), axis=1)
            # print("Calculating ConvexHull...")
            hull = ConvexHull(data)
            # print("ConvexHull calculated.")
            upper_line, _ = getUpperLine(hull, data)
            plt.plot(upper_line[:,0], upper_line[:,1], color=model_color, label=model_label, marker=model_marker, markersize=markersize, linewidth=linewidth, markeredgewidth=model_markeredgewidth)
        ax.grid()
        ax.set_xlabel('Trainable parameters')
        ax.set_ylabel('VPT')
        ax.set_ylim(bottom=0)
        if xlims != None:
            ax.set_xlim(xlims)
        # ax.xaxis.major.formatter._useMathText = True
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        NUM_ACC_PRED_STR_SHORT = "NAP" if NUM_ACC_PRED_STR =="num_accurate_pred_050_avg_TEST" else 0
        fig.savefig(fig_path + '/{:}_2_NUM_TRAINPARAMS_RDIM{:}.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM), bbox_inches='tight')
        lgd = ax.legend(loc="upper left", bbox_to_anchor=(1,1))
        fig.savefig(fig_path + '/{:}_2_NUM_TRAINPARAMS_RDIM{:}_LEGEND.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()




        fig, ax = plt.subplots(1)
        for MODEL in range(N):
            model_name = model_names_str[MODEL]
            model_color = model_color_str[MODEL]
            model_marker = model_marker_str[MODEL]
            model_markeredgewidth = model_marker_width_str[MODEL]
            model_label = model_label_str[MODEL]
            valid_time_vec = valid_time_vecs[model_name]
            n_TRAINPARAMS_vec = n_TRAINPARAMS_vecs[model_name]
            n_MODELPARAMS_vec = n_MODELPARAMS_vecs[model_name]
            n_MODELPARAMS_vec = np.reshape(n_MODELPARAMS_vec, (-1,1))
            valid_time_vec = np.reshape(valid_time_vec, (-1,1))
            data = np.concatenate((n_MODELPARAMS_vec, valid_time_vec), axis=1)
            # print("Calculating ConvexHull...")
            hull = ConvexHull(data)
            # print("ConvexHull calculated.")
            upper_line, _ = getUpperLine(hull, data)
            plt.plot(upper_line[:,0], upper_line[:,1], color=model_color, label=model_label, marker=model_marker, markersize=markersize, linewidth=linewidth, markeredgewidth=model_markeredgewidth)
        ax.grid()
        ax.set_xlabel('Model parameters')
        ax.set_ylabel('VPT')
        ax.set_ylim(bottom=0)
        if xlims != None:
            ax.set_xlim(xlims)
        NUM_ACC_PRED_STR_SHORT = "NAP" if NUM_ACC_PRED_STR =="num_accurate_pred_050_avg_TEST" else 0
        # ax.xaxis.major.formatter._useMathText = True
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        fig.savefig(fig_path + '/{:}_2_NUM_MODELPARAMS_RDIM{:}.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM), bbox_inches='tight')
        lgd = ax.legend(loc="upper left", bbox_to_anchor=(1,1))
        fig.savefig(fig_path + '/{:}_2_NUM_MODELPARAMS_RDIM{:}_LEGEND.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()







