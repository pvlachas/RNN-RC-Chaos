#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import pickle
import glob, os
import numpy as np
import argparse
import time

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


if system_name=="Lorenz3D":
    def getAllModelsTrainDict(saving_path):
        modeldict = {}
        for subdir, dirs, files in os.walk(saving_path):
            for filename in files:
                # print(filename)
                if filename == "train.txt":
                    filedir = os.path.join(subdir, filename)
                    # print(filedir)
                    with open(filedir, 'r') as file_object:  
                        for line in file_object:
                            # print(line)
                            modeldict=parseLineDict(line, filename, modeldict)    
        return modeldict

    def getAllModelsTestDict(saving_path):
        modeldict = {}
        for subdir, dirs, files in os.walk(saving_path):
            for filename in files:
                # print(filename)
                if filename == "test.txt":
                    filedir = os.path.join(subdir, filename)
                    # print(filedir)
                    with open(filedir, 'r') as file_object:  
                        for line in file_object:
                            # print(line)
                            modeldict=parseLineDict(line, filename, modeldict)    
        return modeldict


    model_names_str = [
    "RNN-esn(.*)-RDIM_(.*)-N_used_(.*)-SIZE_(.*)-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)(.*)-IPL_(.*)-REG_(.*)-WID_0",
    "RNN-gru-RDIM_(.*)-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
    "RNN-lstm-RDIM_(.*)-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
    "RNN-unitary-RDIM_(.*)-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
    ]


else:

    def getAllModelsTestDict(saving_path):
        os.chdir(saving_path)
        filename='./test.txt'
        modeldict = {}
        with open(filename, 'r') as file_object:  
            for line in file_object:
                # print(line)
                modeldict=parseLineDict(line, filename, modeldict)
        filename='./test_old.txt'
        try:
            with open(filename, 'r') as file_object:  
                for line in file_object:
                    # print(line)
                    modeldict=parseLineDict(line, filename, modeldict)
        except FileNotFoundError: 
            pass
        filename='./test_OLD.txt'
        try:
            with open(filename, 'r') as file_object:  
                for line in file_object:
                    # print(line)
                    modeldict=parseLineDict(line, filename, modeldict)
        except FileNotFoundError: 
            pass
        return modeldict

    def getAllModelsTrainDict(saving_path):
        os.chdir(saving_path)
        modeldict = {}
        filename='./train_old.txt'
        try:
            with open(filename, 'r') as file_object:  
                for line in file_object:
                    # print(line)
                    modeldict=parseLineDict(line, filename, modeldict)
        except FileNotFoundError: 
            pass
        filename='./train_OLD.txt'
        try:
            with open(filename, 'r') as file_object:  
                for line in file_object:
                    # print(line)
                    modeldict=parseLineDict(line, filename, modeldict)
        except FileNotFoundError: 
            pass
        filename='./train.txt'
        with open(filename, 'r') as file_object:  
            for line in file_object:
                # print(line)
                modeldict=parseLineDict(line, filename, modeldict)
        return modeldict


    model_names_str = [
    "GPU-RNN-esn(.*)-RDIM_(.*)-N_used_(.*)-SIZE_(.*)-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)(.*)-IPL_(.*)-REG_(.*)-WID_0",
    "GPU-RNN-gru-RDIM_(.*)-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
    "GPU-RNN-lstm-RDIM_(.*)-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
    "GPU-RNN-unitary-RDIM_(.*)-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
    ]



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


val_results_path = saving_path + global_params.results_dir
train_results_path = saving_path + global_params.model_dir

if system_name == "Lorenz3D":
    dt=1
    lambda1=1
    RDIM_VALUES = [1,2,3]
elif system_name == "Lorenz96_F8GP40R40":
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


# WDITH=0.12
WDITH=0.22
LENGTH=1.5
widths=0.16

N = len(model_names_str)


for NUM_ACC_PRED_STR in ["num_accurate_pred_050_avg_TEST"]:
    fig, ax = plt.subplots(1)
    ax.yaxis.grid() # horizontal lines
    ax.set_axisbelow(True)

    ITER=0-(N-1)/2
    TOTAL_MODELS = len(model_test_dict)
    model_iter = 0
    model_times = []
    time_start_total = time.time()
    for MODEL in range(N):
        model_name = model_names_str[MODEL]
        model_color = model_color_str[MODEL]
        model_marker = model_marker_str[MODEL]
        model_markeredgewidth = model_marker_width_str[MODEL]
        model_label = model_label_str[MODEL]

        valid_time_2_rdim = {}
        num_ics_deviating_2_rdim = {}
        num_ics_not_deviating_2_rdim = {}
        REGEX = re.compile(model_name)


        for model_name_key in model_test_dict:
            if REGEX.match(model_name_key):
                # print("MATCH!")
                print(model_name_key)
                start = time.time()

                # print(model_test_dict[model_name_key])
                valid_time = model_test_dict[model_name_key][NUM_ACC_PRED_STR]*dt*lambda1
                mni = model_name_key.find("-RDIM_")
                try:
                    rdim = int(model_name_key[mni+6:mni+8])
                except ValueError:
                    rdim = int(model_name_key[mni+6:mni+7])

                val_file = val_results_path + model_name_key + "/results.pickle"
                train_file = train_results_path + model_name_key + "/data.pickle"
                # print("Loading validation results...")
                val_result = pickle.load(open(val_file, "rb" ))
                predictions_all = val_result["predictions_all_TEST"]

                if system_name == "Lorenz3D":
                    truths_all = val_result["targets_all_TEST"]
                else:
                    truths_all = val_result["truths_all_TEST"]
                # print(val_result)

                # print("Loading training results...")
                train_result = pickle.load(open(train_file, "rb" ))
                scaler = train_result["scaler"]
                # time.sleep(0.1)

                num_ics_deviating, num_ics_not_deviating = getNumberOfDivergentTrajectories(truths_all, predictions_all, scaler.data_std)
                end = time.time()

                time_model = end - start
                model_times.append(time_model)
                # print("Time per model {:}".format(time_model))
                time_model = np.mean(np.array(model_times))

                model_iter = model_iter + 1
                time_covered = time.time() - time_start_total #time_model * model_iter
                time_total = time_model * TOTAL_MODELS
                percent = time_covered/time_total*100

                label = "[ Mean model time {:} - TIME={:} / {:} - {:.2f} %]".format(secondsToTimeStr(time_model), secondsToTimeStr(time_covered), secondsToTimeStr(time_total), percent)
                print(label)
                print("###################################################")


                if str(rdim) in valid_time_2_rdim:
                    valid_time_2_rdim[str(rdim)].append(valid_time)
                    num_ics_deviating_2_rdim[str(rdim)].append(num_ics_deviating)
                    num_ics_not_deviating_2_rdim[str(rdim)].append(num_ics_not_deviating)
                else:
                    valid_time_2_rdim[str(rdim)] = [valid_time]
                    num_ics_deviating_2_rdim[str(rdim)] = [num_ics_deviating]
                    num_ics_not_deviating_2_rdim[str(rdim)] = [num_ics_not_deviating]


        if len(valid_time_2_rdim) != 0:
            rdim_vec = []
            valid_time_all_vec = []
            div_pred_vec_all_vec = []
            dictkeys = valid_time_2_rdim.keys()
            dictkeys = [int(key) for key in dictkeys]
            dictkeys = np.sort(dictkeys)
            for intkey in dictkeys:
                rdim = str(intkey)
                valid_time_vec = valid_time_2_rdim[rdim]
                div_pred_vec = num_ics_deviating_2_rdim[rdim]
                rdim_vec.append(int(rdim))

                valid_time_vec = np.array(valid_time_vec)
                div_pred_vec = np.array(div_pred_vec)
                idx = np.argsort(valid_time_vec)
                valid_time_vec = valid_time_vec[idx][::-1]
                div_pred_vec = div_pred_vec[idx][::-1]

                # print(len(error_metric_vec))
                # PERCENT = 0.5
                PERCENT = 0.9
                valid_time_vec = valid_time_vec[:int(np.ceil(PERCENT*len(valid_time_vec)))]
                div_pred_vec = div_pred_vec[:int(np.ceil(PERCENT*len(div_pred_vec)))]
                valid_time_all_vec.append(valid_time_vec)
                div_pred_vec_all_vec.append(div_pred_vec)


            pos = np.arange(len(rdim_vec))*LENGTH+ITER*WDITH
            pos = list(pos)


            vp = ax.violinplot([div_pred_vec_all_vec[0]], [pos[0]], points=10, widths=widths, showmeans=True, showextrema=True, showmedians=False, bw_method=0.5)

            for patch in vp['bodies']:
                patch.set_color(model_color)

            for partname in ('cbars','cmins','cmaxes','cmeans'):
            # for partname in ('cbars','cmins','cmaxes','cmedians','cmeans')::
                vpi = vp[partname]
                vpi.set_edgecolor(model_color)
                vpi.set_linewidth(2)

            vp = ax.violinplot([div_pred_vec_all_vec[1]], [pos[1]], points=10, widths=widths, showmeans=True, showextrema=True, showmedians=False, bw_method=0.5)

            for patch in vp['bodies']:
                patch.set_color(model_color)
                patch.set_label(model_label)

            for partname in ('cbars','cmins','cmaxes','cmeans'):
                vpi = vp[partname]
                vpi.set_edgecolor(model_color)
                vpi.set_linewidth(2)


            ax.set_xticks(np.arange(len(rdim_vec))*LENGTH)
            ax.set_xticklabels(rdim_vec)
        ITER+=1

    ax.set_xlabel('Reduced order dimension')
    ax.set_ylabel('Divergent predictions')
    ax.set_ylim(bottom=0)
    NUM_ACC_PRED_STR_SHORT = "NAP" if NUM_ACC_PRED_STR =="num_accurate_pred_050_avg_TEST" else 0
    fig.savefig(fig_path + '/DIVERGPRED_BBO_{:}_2_RDIM.pdf'.format(NUM_ACC_PRED_STR_SHORT), bbox_inches='tight')
    lgd = ax.legend(loc="upper left", bbox_to_anchor=(1,1), prop={'size': 24})
    fig.savefig(fig_path + '/DIVERGPRED_BBO_{:}_2_RDIM_LEGEND.pdf'.format(NUM_ACC_PRED_STR_SHORT), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

