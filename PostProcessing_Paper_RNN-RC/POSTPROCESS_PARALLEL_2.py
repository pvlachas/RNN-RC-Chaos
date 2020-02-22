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
matplotlib.rc('text', usetex=True)
angle = 0


# PLOTTING THE TRAINING TIME BAR for KuramotoSivashinskyGP512


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
        xlims = None
        model_names_str = [
        "GPU-RNN-esn-PARALLEL-NG_64-RDIM_512-N_used_100000-SIZE_500-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_8-GIL_8",
        "GPU-RNN-esn-PARALLEL-NG_64-RDIM_512-N_used_100000-SIZE_1000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_8-GIL_8",
        "GPU-RNN-esn-PARALLEL-NG_64-RDIM_512-N_used_100000-SIZE_3000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_8-GIL_8",
        # "GPU-RNN-esn-PARALLEL-NG_64-RDIM_{:d}-N_used_100000-SIZE_6000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_8-GIL_8",
        # "GPU-RNN-esn-PARALLEL-NG_64-RDIM_{:d}-N_used_100000-SIZE_12000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_8-GIL_8",
        "GPU-RNN-gru-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_80-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        "GPU-RNN-gru-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_100-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        "GPU-RNN-gru-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_120-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        "GPU-RNN-lstm-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_80-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        "GPU-RNN-lstm-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_100-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        "GPU-RNN-lstm-PARALLEL-NG_64-RDIM_512-N_used_100000-NUM-LAY_1-SIZE-LAY_120-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_8-GIL_8",
        ]
        
        model_label_str = [
        "RC-500", 
        "RC-1000", 
        "RC-3000", 
        # "RC-6000", 
        # "RC-12000", 
        "GRU-80",
        "GRU-100",
        "GRU-120",
        "LSTM-80",
        "LSTM-100",
        "LSTM-120",
        ]
        model_hatch_str = [
        "", 
        "", 
        "", 
        # "", 
        "//", 
        "//", 
        "//", 
        "\\",
        "\\",
        "\\",
        ]

        model_color_str = [
        "blue", 
        "blue", 
        "blue", 
        # "blue", 
        "green",
        "green",
        "green",
        "red",
        "red",
        "red",
        ]
    elif system_name == "Lorenz96_F8GP40":
        dt=0.01
        lambda1=1.68
        RDIM_VALUES = [40]
        xlims = None
        model_names_str = [
        "GPU-RNN-esn-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-SIZE_1000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-esn-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-SIZE_3000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-esn-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-SIZE_6000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-esn-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-SIZE_12000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)-IPL_(.*)-REG_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-gru-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_100-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-gru-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_250-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-gru-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_500-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-lstm-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_100-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-lstm-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_250-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        "GPU-RNN-lstm-PARALLEL-NG_(.*)-RDIM_40-N_used_100000-NUM-LAY_1-SIZE-LAY_500-ACT_tanh-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-GS_(.*)-GIL_4",
        ]
        model_label_str = [
        "RC-1000", 
        "RC-3000", 
        "RC-6000", 
        "RC-12000", 
        "GRU-100",
        "GRU-250",
        "GRU-500",
        "LSTM-100",
        "LSTM-250",
        "LSTM-500",
        ]

        model_hatch_str = [
        "", 
        "", 
        "", 
        "", 
        "//", 
        "//", 
        "//", 
        "\\",
        "\\",
        "\\",
        ]

        model_color_str = [
        "blue", 
        "blue", 
        "blue", 
        "blue", 
        "green",
        "green",
        "green",
        "red",
        "red",
        "red",
        ]


    for NUM_ACC_PRED_STR in ["num_accurate_pred_050_avg_TEST"]:

        fig, ax = plt.subplots(1)
        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.set_axisbelow(True)

        training_time_vec = []
        labels_vec = []
        model_color_vec = []
        model_hatch_vec = []

        for MODEL in range(len(model_names_str)):
            model_name = model_names_str[MODEL]
            model_color = model_color_str[MODEL]
            model_hatch = model_hatch_str[MODEL]
            model_label = model_label_str[MODEL]

            valid_time_ = -1
            model_label_ = model_label
            model_color_ = model_color
            model_hatch_ = model_hatch
            train_path_ = None
            REGEX = re.compile(model_name)
            for model_name_key in model_test_dict:
                # print(model_name_key)
                model_train = model_train_dict[model_name_key]
                if REGEX.match(model_name_key):
                    # print("MATCHED!")
                    valid_time = model_test_dict[model_name_key][NUM_ACC_PRED_STR]*dt*lambda1
                    # training_time = model_train_dict[model_name_key]["total_training_time"]
                    if valid_time > valid_time_:
                        valid_time_ = valid_time
                        model_label_ = model_label
                        model_color_ = model_color
                        model_hatch_ = model_hatch
                        train_path_ = train_results_path + model_name_key
                        temp = model_name_key.split("_")
                        NPG = temp[1].split("-")[0]
                        GS = temp[-2].split("-")[0]
                        GIL = temp[-1]
                        TYPE = temp[0].split("-")[2]

            training_time = getMeanTrainingTimeOfParallelModel(train_path_, NPG, GS, GIL, TYPE)
            print(training_time)
            training_time_vec.append(training_time)
            labels_vec.append(model_label_)
            model_color_vec.append(model_color_)
            model_hatch_vec.append(model_hatch_)

        indexes = np.arange(len(training_time_vec))
        print(indexes)
        print(training_time_vec)
        iter_ = 0
        for index in indexes[::-1]:
            # plt.barh(index, training_time_vec[iter_], color=model_color_vec[iter_])
            # plt.barh(index, training_time_vec[iter_], color=model_color_vec[iter_], hatch=model_hatch_vec[iter_], fill=False, ecolor=model_color_vec[iter_], edgecolor=model_color_vec[iter_])
            plt.barh(index, training_time_vec[iter_], color=model_color_vec[iter_], hatch=model_hatch_vec[iter_], ecolor="white", edgecolor="white")
            iter_ = iter_ + 1
            # rects1 = ax.bar(ind - width/2, abs_error_rc, width, label='RC', color=color_str[1], fill=False, hatch='//', ecolor=color_str[1], edgecolor=color_str[1])

        plt.yticks(indexes, labels_vec[::-1], fontsize=16, rotation=0)
        # ax.xaxis.major.formatter._useMathText = True
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel('Training time [s]', fontsize=16)
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
        NUM_ACC_PRED_STR_SHORT = "NAP" if NUM_ACC_PRED_STR =="num_accurate_pred_050_avg_TEST" else 0
        fig.savefig(fig_path + '/P_TRAINTIME_BAR_BBO_{:}.pdf'.format(NUM_ACC_PRED_STR_SHORT), bbox_inches='tight')

        plt.close()









