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
linewidth = 3
markersize = 10
# matplotlib.rc('text', usetex=True)



# python3 POSTPROCESS_11.py --system_name Lorenz3D
# python3 POSTPROCESS_11.py --system_name Lorenz96_F8GP40R40
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


# PLOTTING AVERAGE NRMSE PLOT W.R.T. TIME

val_results_path = saving_path + global_params.results_dir
train_results_path = saving_path + global_params.model_dir

if system_name == "Lorenz3D":
    RDIM_VALUES = [1,2,3]
    # xlims = np.array([100, 500, 1000])*dt*lambda1
    xlims = np.array([0.5, 1, 2, 5, 10])
    ylims = [[-6, 1]]
elif system_name == "Lorenz96_F8GP40R40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [35,40]
    # xlims = np.array([100, 500, 1000])*dt*lambda1
    xlims = np.array([0.5, 1, 2, 5, 10])
    ylims = [[-6, 1]]
elif system_name == "Lorenz96_F10GP40R40":
    dt=0.01
    lambda1=2.27
    RDIM_VALUES = [35,40]
    # xlims = np.array([100, 500, 1000])*dt*lambda1
    xlims = np.array([0.5, 1, 2, 5, 10])
    ylims = [[-6, 1]]
elif system_name == "Lorenz96_F8GP40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [40]
    # xlims = np.array([100, 500, 1000])*dt*lambda1
    xlims = np.array([0.5, 1, 2, 5, 10])
    ylims = [[-6, 1]]

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

for NUM_ACC_PRED_STR in ["num_accurate_pred_050_avg_TEST"]:

    for r in range(len(RDIM_VALUES)):
        RDIM = RDIM_VALUES[r]

        time_vector_list = []
        rmnse_best_list = []
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
            model_markeredgewidth = model_marker_width_str[MODEL]

            valid_time_ = -1
            REGEX = re.compile(model_name)
            for model_name_key in model_test_dict:
                model_train = model_train_dict[model_name_key]
                if REGEX.match(model_name_key):
                    valid_time = model_test_dict[model_name_key][NUM_ACC_PRED_STR]*dt*lambda1
                    if valid_time > valid_time_:
                        valid_time_ = valid_time
                        val_file_ = val_results_path + model_name_key + "/results.pickle"
                        train_file_ = train_results_path + model_name_key + "/data.pickle"

            val_result = pickle.load(open(val_file_, "rb" ))
            predictions_all = val_result["predictions_all_TEST"]
            truths_all = val_result["truths_all_TEST"]
            train_result = pickle.load(open(train_file_, "rb" ))
            scaler = train_result["scaler"]
            rmnse_all=[]
            for ic in range(np.shape(truths_all)[0]):
                target = truths_all[ic]
                prediction = predictions_all[ic]
                rmse, nrmse, _, _, _ = computeErrors(target, prediction, scaler.data_std)
                rmnse_all.append(nrmse)
            rmnse_all = np.array(rmnse_all)
            rmnse_best = np.mean(rmnse_all, axis=0)
            rmnse_best = np.log(rmnse_best)

            time_vector = np.arange(len(rmnse_best))
            # ax.plot(time_vector, rmnse_best, color=model_color, label=model_label)

            time_vector_list.append(time_vector)
            rmnse_best_list.append(rmnse_best)
            model_color_list.append(model_color)
            model_label_list.append(model_label)
            model_marker_list.append(model_marker)
            markersize_list.append(markersize)
            linewidth_list.append(linewidth)
            model_markeredgewidth_list.append(model_markeredgewidth)


        bounds_iter = 0
        for xlim in xlims:
            for ylim in ylims:
                fig, ax = plt.subplots(1)
                for P in range(len(time_vector_list)):
                    num_samples=np.sum(time_vector_list[P]*dt*lambda1<xlim)
                    markevery=int(num_samples/12)
                    markevery=getPrimeNumbers(len(model_names_str), markevery)[P]
                    ax.plot(time_vector_list[P]*dt*lambda1, rmnse_best_list[P], color=model_color_list[P], label=model_label_list[P], marker=model_marker_list[P], markersize=markersize_list[P], linewidth=linewidth_list[P], markeredgewidth=model_markeredgewidth_list[P],markevery=int(markevery))
                ax.grid()
                ax.set_xlabel('$t \, / \, T^{\Lambda_1}$')
                ax.set_ylabel('Log of Average NRMSE')
                ax.set_xlim([0, xlim])
                ax.set_ylim(ylim)
                NUM_ACC_PRED_STR_SHORT = "NAP" if NUM_ACC_PRED_STR =="num_accurate_pred_050_avg_TEST" else 0
                fig.savefig(fig_path + '/LOG_RMNSE_BBO_{:}_2_M_RDIM_{:}_{:}.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM, bounds_iter), bbox_inches='tight')
                plt.close()
                bounds_iter = bounds_iter + 1

        bounds_iter = 0
        for xlim in xlims:
            for ylim in ylims:
                fig, ax = plt.subplots(1)
                for P in range(len(time_vector_list)):
                    num_samples=np.sum(time_vector_list[P]*dt*lambda1<xlim)
                    markevery=int(num_samples/12)
                    markevery=getPrimeNumbers(len(model_names_str), markevery)[P]
                    ax.plot(time_vector_list[P]*dt*lambda1, rmnse_best_list[P], color=model_color_list[P], label=model_label_list[P], marker=model_marker_list[P], markersize=markersize_list[P], linewidth=linewidth_list[P], markeredgewidth=model_markeredgewidth_list[P],markevery=markevery)
                ax.grid()
                ax.set_xlabel('$t \, / \, T^{\Lambda_1}$')
                ax.set_ylabel('Log of Average NRMSE')
                ax.set_xlim([0, xlim])
                ax.set_ylim(ylim)
                lgd = ax.legend(loc="upper left", bbox_to_anchor=(1,1))
                NUM_ACC_PRED_STR_SHORT = "NAP" if NUM_ACC_PRED_STR =="num_accurate_pred_050_avg_TEST" else 0
                fig.savefig(fig_path + '/LOG_RMNSE_BBO_{:}_2_M_RDIM_{:}_{:}_LEGEND.pdf'.format(NUM_ACC_PRED_STR_SHORT, RDIM, bounds_iter), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.close()
                bounds_iter = bounds_iter + 1

