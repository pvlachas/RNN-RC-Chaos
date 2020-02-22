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


from itertools import cycle

# PLOTTING
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from Utils.utils import *

font = {'size'   : 16, 'family':'Times New Roman'}
matplotlib.rc('font', **font)
linewidth = 3
markersize = 10
# matplotlib.rc('text', usetex=True)


from scipy.spatial import ConvexHull

# python3 POSTPROCESS_2.py --system_name Lorenz3D

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



# PLOTTING THE NUM OF ACCURATE PREDICTIONS W.R.T. RAM MEMORY

if system_name == "Lorenz3D":
    RDIM_VALUES = [1,2,3]
    xlims = None
elif system_name == "Lorenz96_F8GP40R40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [35,40]
    xlims = None
elif system_name == "Lorenz96_F10GP40R40":
    dt=0.01
    lambda1=2.27
    RDIM_VALUES = [35,40]
    xlims = None
elif system_name == "Lorenz96_F8GP40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [40]
    xlims = None

for NUM_ACC_PRED_STR in ["num_accurate_pred_050_avg_TEST"]:

    for RDIM in RDIM_VALUES:
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

        N = len(model_names_str)
        # N = 2

        fig, ax = plt.subplots(1)
        for MODEL in range(N):
            model_name = model_names_str[MODEL]
            model_color = model_color_str[MODEL]
            model_marker = model_marker_str[MODEL]
            model_markeredgewidth = model_marker_width_str[MODEL]
            model_label = model_label_str[MODEL]

            memory_vec = []
            valid_time_vec = []

            REGEX = re.compile(model_name)
            for model_name_key in model_test_dict:
                # print(model_name_key)
                model_train = model_train_dict[model_name_key]
                # print(model_train)
                if REGEX.match(model_name_key):
                    # print(model_test_dict[model_name_key])
                    valid_time = model_test_dict[model_name_key][NUM_ACC_PRED_STR]*dt*lambda1
                    memory = model_train_dict[model_name_key]["memory"]
                    memory_vec.append(memory)
                    valid_time_vec.append(valid_time)

            memory_vec = np.reshape(memory_vec, (-1,1))
            valid_time_vec = np.reshape(valid_time_vec, (-1,1))
            data = np.concatenate((memory_vec, valid_time_vec), axis=1)
                
            # print("Calculating ConvexHull...")
            hull = ConvexHull(data)
            # print("ConvexHull calculated.")
            upper_line, all_lines = getUpperLine(hull, data)

            plt.plot(upper_line[:,0], upper_line[:,1], color=model_color, label=model_label, marker=model_marker, markersize=markersize, linewidth=linewidth, markeredgewidth=model_markeredgewidth)
            # plt.fill(data[hull.vertices,0], data[hull.vertices,1], color=model_color, alpha=0.25, label=model_label)
            # plt.plot(all_lines[:,0], all_lines[:,1], "--", color=model_color)

        ax.grid()
        ax.set_xlabel('RAM Memory [MB]')
        ax.set_ylabel('VPT')
        ax.set_ylim(bottom=0)
        # ax.xaxis.major.formatter._useMathText = True
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        # plt.show()
        NUM_ACC_PRED_STR_SHORT = "NAP" if NUM_ACC_PRED_STR =="num_accurate_pred_050_avg_TEST" else 0
        fig.savefig(fig_path + '/{:}_2_RAM_RDIM{:}.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM), bbox_inches='tight')
        lgd = ax.legend(loc="upper left", bbox_to_anchor=(1,1))
        fig.savefig(fig_path + '/{:}_2_RAM_RDIM{:}_LEGEND.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        # print(ark)





