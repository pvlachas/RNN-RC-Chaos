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
import matplotlib.pyplot as plt
import re
from Utils.utils import *

font = {'size'   : 16, 'family':'Times New Roman'}
matplotlib.rc('font', **font)
linewidth = 2
markersize = 8
angle = 0


# PLOTTING THE OVERFITTING SCATTERING PLOT for KuramotoSivashinskyGP512


# python3 POSTPROCESS_14_parallel.py --system_name KuramotoSivashinskyGP512


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--system_name", help="system", type=str, required=True)
    parser.add_argument("--Experiment_Name", help="system", type=str, required=False, default=None)
    args = parser.parse_args()
    system_name = args.system_name
    Experiment_Name = args.Experiment_Name

    if Experiment_Name is None or Experiment_Name=="None":
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

    val_results_path = saving_path + global_params.results_dir
    train_results_path = saving_path + global_params.model_dir



    if system_name == "KuramotoSivashinskyGP512":
        dt=0.25
        lambda1=0.094        
        RDIM_VALUES = [512]
        model_names_str = [
        "GPU-RNN-esn-PARALLEL-NG_64-RDIM_512-N_used_100000-SIZE_(.*)-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_8-GIL_8",
        "GPU-RNN-gru-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_(.*)-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        "GPU-RNN-lstm-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_(.*)-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        ]
        model_label_str = [
        "RC", 
        "GRU",
        "LSTM",
        "Unit",
        ]

    elif system_name == "Lorenz96_F8GP40":
        dt=0.01
        lambda1=1.68
        RDIM_VALUES = [40]
        xlims = None
        model_names_str = [
        "GPU-RNN-esn-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-SIZE_(.*)-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-gru-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_(.*)-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-lstm-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_(.*)-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        ]
        model_label_str = [
        "RC", 
        "GRU",
        "LSTM",
        "Unit",
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
    MARKERSIZE=2**6
    model_marker_size_str = [
    MARKERSIZE, 
    MARKERSIZE, 
    MARKERSIZE, 
    MARKERSIZE,
    ]

    for NUM_ACC_PRED_STR in ["num_accurate_pred_050_avg"]:

        for r in range(len(RDIM_VALUES)):
            RDIM = RDIM_VALUES[r]

            valid_times_test_list = []
            valid_times_train_list = []
            model_label_list = []
            model_marker_list = []
            markersize_list = []
            model_color_list = []
            model_markersize_list = []
            linewidth_list = []

            for MODEL in range(len(model_names_str)):
                model_name = model_names_str[MODEL].format(RDIM)
                model_label = model_label_str[MODEL]
                model_color = model_color_str[MODEL]
                model_marker = model_marker_str[MODEL]
                model_markersize = model_marker_size_str[MODEL]

                valid_time_test=[]
                valid_time_train=[]

                valid_time_ = -1
                REGEX = re.compile(model_name)
                for model_name_key in model_test_dict:
                    model_train = model_train_dict[model_name_key]
                    if REGEX.match(model_name_key):
                        valid_time_test_ = model_test_dict[model_name_key][NUM_ACC_PRED_STR+"_TEST"]*dt*lambda1
                        valid_time_train_ = model_test_dict[model_name_key][NUM_ACC_PRED_STR+"_TRAIN"]*dt*lambda1
                        valid_time_test.append(valid_time_test_)
                        valid_time_train.append(valid_time_train_)

                valid_times_test_list.append(valid_time_test)
                valid_times_train_list.append(valid_time_train)
                model_color_list.append(model_color)
                model_label_list.append(model_label)
                model_marker_list.append(model_marker)
                linewidth_list.append(linewidth)
                model_markersize_list.append(model_markersize)

            fig, ax = plt.subplots(1)
            for P in range(len(model_color_list)):
                if model_marker_list[P] in ["s","o","d"]:
                    plt.scatter(valid_times_train_list[P], valid_times_test_list[P], label=model_label_list[P], marker=model_marker_list[P], edgecolors=model_color_list[P], linewidth=linewidth_list[P], facecolors='none', s=model_markersize_list[P])
                else:
                    plt.scatter(valid_times_train_list[P], valid_times_test_list[P], label=model_label_list[P], marker=model_marker_list[P], color=model_color_list[P], linewidth=linewidth_list[P], s=model_markersize_list[P])
            LIM=np.max(np.max(np.array(valid_times_train_list)))
            LIM=round(LIM * 2) / 2
            plt.plot(LIM*np.linspace(0,1,10), LIM*np.linspace(0,1,10), 'k--')
            ax.grid()
            ax.set_axisbelow(True)
            NUM_ACC_PRED_STR_SHORT = "NAP" if NUM_ACC_PRED_STR=="num_accurate_pred_050_avg" else 0
            ax.set_xlabel('VPT in train dataset')
            ax.set_ylabel('VPT in test dataset')
            ax.set_xlim([0, LIM])
            ax.set_ylim([0, LIM])
            fig.savefig(fig_path + '/P_OFSP_BBO_{:}_RDIM_{:}.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM), bbox_inches='tight')
            lgd = ax.legend(loc="upper left", bbox_to_anchor=(1,1))
            fig.savefig(fig_path + '/P_OFSP_BBO_{:}_RDIM_{:}_LEGEND.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM), bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()


