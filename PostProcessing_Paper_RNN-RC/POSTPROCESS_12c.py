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



# python3 POSTPROCESS_12b.py --system_name Lorenz96_F8GP40R40 --Experiment_Name Experiment_Daint_Large

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


# PLOTTING SPECTRUM OF THE BEST MODEL


val_results_path = saving_path + global_params.results_dir
train_results_path = saving_path + global_params.model_dir


if system_name == "Lorenz3D":
    RDIM_VALUES = [1,2,3]
    # ylims = [None, [0,2], [0,5]]
    # xlims = [None, [-20,20]]
    ylims = [None]
    xlims = [None]
elif system_name == "Lorenz96_F8GP40R40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [35,40]
    ylims = [None]
    xlims = [None]
elif system_name == "Lorenz96_F10GP40R40":
    dt=0.01
    lambda1=2.27
    RDIM_VALUES = [35,40]
    xlims = [None]
    ylims = [None]
elif system_name == "Lorenz96_F8GP40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [40]
    xlims = [None]
    ylims = [None]

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


WDITH=0.22
LENGTH=1.5
widths=0.16

N = len(model_names_str)

for CRITERION in ["num_accurate_pred_050_avg_TEST", "error_freq_TEST"]:
    fig, ax = plt.subplots(1)
    ax.yaxis.grid() # horizontal lines
    ax.set_axisbelow(True)

    for r in range(len(RDIM_VALUES)):
        ITER=0-(N-1)/2
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
            error_metric_vec = []
            val_file_vec = []
            train_file_vec = []
            vpt_vec = []
            # error_metric_ = 1e10
            # error_metric_ = 1e10
            REGEX = re.compile(model_name)
            for model_name_key in model_test_dict:
                model_train = model_train_dict[model_name_key]
                if REGEX.match(model_name_key):
                    # print(model_name_key)

                    error_metric = model_test_dict[model_name_key][CRITERION]
                    if CRITERION=="num_accurate_pred_050_avg_TEST":
                        error_metric = (-error_metric)

                    error_metric_vec.append(error_metric)
                    val_file_vec.append(val_results_path + model_name_key + "/results.pickle")
                    train_file_vec.append(train_results_path + model_name_key + "/data.pickle")
                    vpt_vec.append(model_test_dict[model_name_key]["num_accurate_pred_050_avg_TEST"]*dt*lambda1)

            error_metric_vec = np.array(error_metric_vec)
            val_file_vec = np.array(val_file_vec)
            vpt_vec = np.array(vpt_vec)
            idx = np.argsort(error_metric_vec)
            error_metric_vec = error_metric_vec[idx]
            val_file_vec = val_file_vec[idx]
            vpt_vec = vpt_vec[idx]
            # print(error_metric_vec)
            # print(idx)
            # print(error_metric_vec[0])
            # print(error_metric_vec[10])
            # PERCENT = 0.02


            PERCENT=0.9
            N_MODELS = int(np.ceil(PERCENT*len(error_metric_vec)))

            # N_MODELS = 10 # int(np.ceil(0.02*len(error_metric_vec)))

            error_metric_vec = error_metric_vec[:N_MODELS]
            val_file_vec = val_file_vec[:N_MODELS]
            vpt_vec = vpt_vec[:N_MODELS]
            # print(vpt_vec)
        
            print("NUMBER OF MODELS:")
            print(len(val_file_vec))
            print(model_name)
            VPT_vec = vpt_vec

            pos = np.arange(len(RDIM_VALUES))*LENGTH+ITER*WDITH
            pos = list(pos)


            vp = ax.violinplot(VPT_vec, [pos[r]], points=10, widths=widths, showmeans=True, showextrema=True, showmedians=False, bw_method=0.5)

            for patch in vp['bodies']:
                patch.set_color(model_color)
                if r==1:
                    patch.set_label(model_label)


            for partname in ('cbars','cmins','cmaxes','cmeans'):
            # for partname in ('cbars','cmins','cmaxes','cmedians','cmeans'):
                vpi = vp[partname]
                vpi.set_edgecolor(model_color)
                vpi.set_linewidth(2)


            ITER+=1
    ax.set_xlabel('Reduced order dimension')
    ax.set_ylabel('VPT')
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(len(RDIM_VALUES))*LENGTH)
    ax.set_xticklabels(RDIM_VALUES)
    CRITERION_SHORT = "NAP" if CRITERION =="num_accurate_pred_050_avg_TEST" else "FREQERROR"
    fig.savefig(fig_path + '/VPT_BBO_{:}_RDIM_MDLS.pdf'.format(CRITERION_SHORT), bbox_inches='tight')
    lgd = ax.legend(loc="upper left", bbox_to_anchor=(1,1), prop={'size': 24})
    fig.savefig(fig_path + '/VPT_BBO_{:}_RDIM_MDLS_LEGEND.pdf'.format(CRITERION_SHORT), bbox_extra_artists=(lgd,), bbox_inches='tight')


