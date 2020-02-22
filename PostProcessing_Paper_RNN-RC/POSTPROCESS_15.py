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
# matplotlib.rc('text', usetex=True)

# matplotlib.rc('text', usetex=True)


# python3 POSTPROCESS_10.py --system_name Lorenz3D
# python3 POSTPROCESS_10.py --system_name Lorenz96_F8GP40R40
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



# PLOTTING OVERFITTING PLOT - VALID TIME IN TEST SET W.R.T. VALID TIME IN TRAINING SET


val_results_path = saving_path + global_params.results_dir
train_results_path = saving_path + global_params.model_dir


if system_name == "Lorenz96_F8GP40R40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [35,40]
elif system_name == "Lorenz96_F10GP40R40":
    dt=0.01
    lambda1=2.27
    RDIM_VALUES = [35,40]
elif system_name == "Lorenz96_F8GP40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [40]

model_names_str = [
"GPU-RNN-esn(.*)-RDIM_{:d}-N_used_(.*)-SIZE_(.*)-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)(.*)-IPL_(.*)-REG_(.*)-WID_0",
"GPU-RNN-gru-RDIM_{:d}-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
"GPU-RNN-lstm-RDIM_{:d}-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
"GPU-RNN-unitary-RDIM_{:d}-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
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
model_label_str = [
"RC", 
"GRU",
"LSTM",
"Unit",
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

        ax.set_xlabel('VPT in train dataset')
        ax.set_ylabel('VPT in test dataset')
        ax.set_xlim([0, LIM])
        ax.set_ylim([0, LIM])
        NUM_ACC_PRED_STR_SHORT = "NAP" if NUM_ACC_PRED_STR =="num_accurate_pred_050_avg_TEST" else 0
        fig.savefig(fig_path + '/OverffitingScatterplot_BBO_{:}_RDIM_{:}.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM), bbox_inches='tight')
        lgd = ax.legend(loc="upper left", bbox_to_anchor=(1,1))
        fig.savefig(fig_path + '/OverffitingScatterplot_BBO_{:}_RDIM_{:}_LEGEND.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()


