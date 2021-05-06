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
markersize = 10



# python3 POSTPROCESS_12.py --system_name Lorenz3D
# python3 POSTPROCESS_12.py --system_name Lorenz96_F8GP40R40
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--system_name", help="system", type=str, required=True)
    parser.add_argument("--Experiment_Name", help="Experiment_Name", type=str, required=False, default=None)
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


    # PLOTTING SPECTRUM OF THE BEST MODEL


    val_results_path = saving_path + global_params.results_dir
    train_results_path = saving_path + global_params.model_dir


    if system_name == "KuramotoSivashinskyGP512":
        dt=0.25
        lambda1=0.094    
        RDIM_VALUES = [512]
        xlims = [None, [0, 0.05]]
        ylims = [None, [-16, -9]]

        model_names_str = [
        "GPU-RNN-esn-PARALLEL-NG_64-RDIM_512-N_used_100000-SIZE_500-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_8-GIL_8",
        "GPU-RNN-esn-PARALLEL-NG_64-RDIM_512-N_used_100000-SIZE_1000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_8-GIL_8",
        "GPU-RNN-esn-PARALLEL-NG_64-RDIM_512-N_used_100000-SIZE_3000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_8-GIL_8",
        # "GPU-RNN-esn-PARALLEL-NG_64-RDIM_{:d}-N_used_100000-SIZE_6000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_8-GIL_8",
        "GPU-RNN-gru-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_80-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        # "GPU-RNN-gru-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_100-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        # "GPU-RNN-gru-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_120-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        "GPU-RNN-lstm-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_80-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        # "GPU-RNN-lstm-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_100-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        # "GPU-RNN-lstm-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_120-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        ]
        model_label_str = [
        "RC-500", 
        "RC-1000", 
        "RC-3000", 
        "GRU-80",
        # "GRU-100",
        # "GRU-120",
        "LSTM-80",
        # "LSTM-100",
        # "LSTM-120",
        ]

    elif system_name == "Lorenz96_F8GP40":
        dt=0.01
        lambda1=1.68
        RDIM_VALUES = [40]
        xlims = [None]
        ylims = [None]
        model_names_str = [
        "GPU-RNN-esn-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-SIZE_1000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_(.*)-GIL_4",
        # "GPU-RNN-esn-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-SIZE_3000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-esn-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-SIZE_6000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-esn-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-SIZE_12000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_(.*)-GIL_4",
        # "GPU-RNN-gru-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_100-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        # "GPU-RNN-gru-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_250-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-gru-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_500-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-lstm-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_100-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        # "GPU-RNN-lstm-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_250-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        # "GPU-RNN-lstm-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_500-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        ]
        model_label_str = [
        "RC-1000", 
        # "RC-3000", 
        "RC-6000", 
        "RC-12000", 
        # "GRU-100",
        # "GRU-250",
        "GRU-500",
        "LSTM-100",
        # "LSTM-250",
        # "LSTM-500",
        ]
        # model_hatch_str = [
        # "", 
        # "", 
        # "", 
        # "//", 
        # "//", 
        # "//", 
        # "\\",
        # "\\",
        # "\\",
        # ]
        # model_color_str = [
        # "blue", 
        # "blue", 
        # "blue", 
        # "green",
        # "green",
        # "green",
        # "red",
        # "red",
        # "red",
        # # "orange",
        # # "blueviolet",
        # # "black",
        # # "cornflowerblue",
        # ]


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
    ">",
    ">",
    ">",
    ">",
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
    2,
    2,
    2,
    2,
    2,
    ]

    for NUM_ACC_PRED_STR in ["num_accurate_pred_050_avg_TEST"]:

        for r in range(len(RDIM_VALUES)):
            RDIM = RDIM_VALUES[r]

            freq_pred_list = []
            sp_pred_list = []
            model_color_list = []
            model_label_list = []
            model_marker_list = []
            markersize_list = []
            linewidth_list = []
            model_markeredgewidth_list = []

            for MODEL in range(len(model_names_str)):
                model_name = model_names_str[MODEL].format(RDIM)
                model_label = model_label_str[MODEL]
                model_color = model_color_str[MODEL]
                model_marker = model_marker_str[MODEL]
                image_name = model_label_str[MODEL]
                model_markeredgewidth = model_marker_width_str[MODEL]


                valid_time_ = -1
                val_file_ = None
                REGEX = re.compile(model_name)
                for model_name_key in model_test_dict:
                    model_train = model_train_dict[model_name_key]
                    if REGEX.match(model_name_key):
                        # print(model_name_key)
                        valid_time = model_test_dict[model_name_key][NUM_ACC_PRED_STR]*dt*lambda1
                        if valid_time > valid_time_:
                            valid_time_ = valid_time
                            val_file_ = val_results_path + model_name_key + "/results.pickle"

                if val_file_ is not None:
                    val_result = pickle.load(open(val_file_, "rb" ))
                    sp_true_TEST = np.array(val_result["sp_true_TEST"])
                    sp_pred_TEST = np.array(val_result["sp_pred_TEST"])
                    freq_pred_TEST = np.array(val_result["freq_pred_TEST"])
                    freq_true_TEST = np.array(val_result["freq_true_TEST"])

                    # ax.plot(freq_pred_TEST, sp_pred_TEST, color=model_color, label=model_label)

                    freq_pred_list.append(freq_pred_TEST)
                    sp_pred_list.append(sp_pred_TEST)
                    model_color_list.append(model_color)
                    model_label_list.append(model_label)
                    model_marker_list.append(model_marker)
                    markersize_list.append(markersize)
                    linewidth_list.append(linewidth)
                    model_markeredgewidth_list.append(model_markeredgewidth)

            # print(freq_pred_list)
            bounds_iter = 0
            for xlim in xlims:
                for ylim in ylims:
                    fig, ax = plt.subplots(1)
                    for P in range(len(freq_pred_list)):
                        if xlim != None:
                            num_samples=np.sum(np.logical_and(np.array(freq_pred_list[P])<=xlim[1], np.array(freq_pred_list[P])>=xlim[0]))
                        else:
                            num_samples=np.shape(freq_pred_list[P])[0]
                        markevery=int(num_samples/12)
                        markevery=getPrimeNumbers(len(model_names_str), markevery)[P]
                        ax.plot(freq_pred_list[P], sp_pred_list[P], color=model_color_list[P], label=model_label_list[P], marker=model_marker_list[P], markersize=markersize_list[P], linewidth=linewidth_list[P], markeredgewidth=model_markeredgewidth_list[P],markevery=markevery)
                    ax.plot(freq_true_TEST, sp_true_TEST, '--', color="black", label="Truth", linewidth=3)
                    ax.grid()
                    ax.set_xlabel('Frequency [Hz]')
                    ax.set_ylabel('Power Spectrum [dB]')
                    if xlim != None: ax.set_xlim(xlim)
                    if ylim != None: ax.set_ylim(ylim)
                    NUM_ACC_PRED_STR_SHORT = "NAP" if NUM_ACC_PRED_STR =="num_accurate_pred_050_avg_TEST" else 0
                    fig.savefig(fig_path + '/P_POWSPEC_BBO_{:}_M_RDIM_{:}_{:}.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM, bounds_iter), bbox_inches='tight')
                    plt.close()
                    bounds_iter = bounds_iter + 1

            bounds_iter = 0
            for xlim in xlims:
                for ylim in ylims:
                    fig, ax = plt.subplots(1)
                    for P in range(len(freq_pred_list)):
                        if xlim != None:
                            num_samples=np.sum(np.logical_and(np.array(freq_pred_list[P])<=xlim[1], np.array(freq_pred_list[P])>=xlim[0]))
                        else:
                            num_samples=np.shape(freq_pred_list[P])[0]
                        markevery=int(num_samples/12)
                        markevery=getPrimeNumbers(len(model_names_str), markevery)[P]
                        ax.plot(freq_pred_list[P], sp_pred_list[P], color=model_color_list[P], label=model_label_list[P], marker=model_marker_list[P], markersize=markersize_list[P], linewidth=linewidth_list[P], markeredgewidth=model_markeredgewidth_list[P],markevery=markevery)
                    ax.plot(freq_true_TEST, sp_true_TEST, '--', color="black", label="Truth", linewidth=3)
                    ax.grid()
                    ax.set_xlabel('Frequency [Hz]')
                    ax.set_ylabel('Power Spectrum [dB]')
                    if xlim != None: ax.set_xlim(xlim)
                    if ylim != None: ax.set_ylim(ylim)
                    lgd = ax.legend(loc="upper left", bbox_to_anchor=(1,1))
                    NUM_ACC_PRED_STR_SHORT = "NAP" if NUM_ACC_PRED_STR =="num_accurate_pred_050_avg_TEST" else 0
                    fig.savefig(fig_path + '/P_POWSPEC_BBO_{:}_M_RDIM_{:}_{:}_LEGEND.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM, bounds_iter), bbox_extra_artists=(lgd,), bbox_inches='tight')
                    plt.close()
                    bounds_iter = bounds_iter + 1


