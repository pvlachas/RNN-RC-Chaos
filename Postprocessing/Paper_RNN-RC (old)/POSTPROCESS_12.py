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
# python3 POSTPROCESS_12.py --system_name Lorenz96_F8GP40R40 --Experiment_Name="Experiment_Daint_Large"

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


for CRITERION in ["error_freq_TEST", "num_accurate_pred_050_avg_TEST"]:

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

            error_metric_vec = []
            val_file_vec = []
            train_file_vec = []
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

            error_metric_vec = np.array(error_metric_vec)
            val_file_vec = np.array(val_file_vec)
            idx = np.argsort(error_metric_vec)
            error_metric_vec = error_metric_vec[idx]
            val_file_vec = val_file_vec[idx]
            # print(idx)
            # print(error_metric_vec[0])
            # print(error_metric_vec[10])

            PERCENT = 0.2
            N_MODELS = int(np.ceil(PERCENT*len(error_metric_vec)))

            # N_MODELS = 2 # int(np.ceil(0.02*len(error_metric_vec)))

            error_metric_vec = error_metric_vec[:N_MODELS]
            val_file_vec = val_file_vec[:N_MODELS]
            train_result = pickle.load(open(train_file_vec[0], "rb" ))
            scaler = train_result["scaler"]

            for i in range(len(val_file_vec)):
                val_file_ = val_file_vec[i]
                val_result = pickle.load(open(val_file_, "rb" ))
                predictions_all = val_result["predictions_all_TEST"]
                truths_all = val_result["truths_all_TEST"]

                sp_pred_TEST, freq_pred_TEST, _, _ = computeSpectrumPostProcessing(truths_all, predictions_all, scaler.data_std, dt)

                val_result["sp_pred_TEST"] = sp_pred_TEST
                val_result["freq_pred_TEST"] = freq_pred_TEST

            val_result_best = pickle.load(open(val_file_vec[-1], "rb" ))

            sp_true_TEST = np.array(val_result_best["sp_true_TEST"])
            freq_true_TEST = np.array(val_result_best["freq_true_TEST"])
            sp_pred_TEST = np.array(val_result_best["sp_pred_TEST"])
            freq_pred_TEST = np.array(val_result_best["freq_pred_TEST"])
            if freq_pred_TEST is None:
                print("#### NOT A SINGLE NON DEVIATING IC WAS FOUND FOR MODEL.\n{:}".format(model_name))

            freq_pred_list.append(freq_pred_TEST)
            sp_pred_list.append(sp_pred_TEST)
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
                CRITERION_SHORT = "NAP" if CRITERION =="num_accurate_pred_050_avg_TEST" else "FREQERROR"
                fig.savefig(fig_path + '/POWSPEC_BBO_{:}_M_RDIM_{:}_{:}.pdf'.format(CRITERION_SHORT, RDIM, bounds_iter), bbox_inches='tight')
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
                CRITERION_SHORT = "NAP" if CRITERION =="num_accurate_pred_050_avg_TEST" else "FREQERROR"
                fig.savefig(fig_path + '/POWSPEC_BBO_{:}_M_RDIM_{:}_{:}_LEGEND.pdf'.format(CRITERION_SHORT, RDIM, bounds_iter), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.close()
                bounds_iter = bounds_iter + 1



