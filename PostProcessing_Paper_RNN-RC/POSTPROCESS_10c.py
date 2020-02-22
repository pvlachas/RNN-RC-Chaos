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
linewidth = 4
markersize = 10
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



# PLOTTING AVERAGE NRMSE PLOT W.R.T. TIME


val_results_path = saving_path + global_params.results_dir
train_results_path = saving_path + global_params.model_dir


if system_name == "Lorenz3D":
    RDIM_VALUES = [1,2,3]
    NSTEPS=int(5/lambda1/dt)
elif system_name == "Lorenz96_F8GP40R40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [35,40]
    T_max=6
    NSTEPS=int(np.ceil(T_max/lambda1/dt))+1
elif system_name == "Lorenz96_F10GP40R40":
    dt=0.01
    lambda1=2.27
    RDIM_VALUES = [35,40]
    T_max=6
    NSTEPS=int(np.ceil(T_max/lambda1/dt))+1
elif system_name == "Lorenz96_F8GP40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [40]
    T_max=6
    NSTEPS=int(np.ceil(T_max/lambda1/dt))+1


model_names_str = [
"GPU-RNN-gru-RDIM_{:d}-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
"GPU-RNN-lstm-RDIM_{:d}-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
"GPU-RNN-esn(.*)-RDIM_{:d}-N_used_(.*)-SIZE_6000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)(.*)-IPL_(.*)-REG_(.*)-WID_0",
"GPU-RNN-esn(.*)-RDIM_{:d}-N_used_(.*)-SIZE_9000-D_(.*)-RADIUS_(.*)-SIGMA_(.*)-DL_(.*)(.*)-IPL_(.*)-REG_(.*)-WID_0",
"GPU-RNN-unitary-RDIM_{:d}-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",
]

"GPU-RNN-gru-RDIM_{:d}-N_used_(.*)-NUM-LAY_(.*)-SIZE-LAY_(.*)-ACT_(.*)-ISH_statefull-SL_(.*)-PL_(.*)-LR_(.*)-DKP_(.*)-ZKP_(.*)-HSPL_(.*)-IPL_(.*)-(.*)WID_0",




model_color_str = [
"green",
"red",
"blue", 
"orange",
"blueviolet",
"black",
"cornflowerblue",
]
model_marker_str = [
"x", 
"o",
"s", 
"d",
"*", 
"<",
">",
]
model_marker_width_str = [
2, 
2, 
3,
2,
2,
2,
2,
]
model_label_str = [
"GRU",
"LSTM",
"RC", 
"RC", 
"UNIT",
]

for CRITERION in ["error_freq_TEST", "num_accurate_pred_050_avg_TEST"]:
        for r in range(len(RDIM_VALUES)):
            RDIM = RDIM_VALUES[r]

            rmnse_all_list = []
            prediction_all_list = []
            truth_all_list = []
            model_color_list = []
            model_label_list = []
            model_marker_list = []
            markersize_list = []
            linewidth_list = []
            model_markeredgewidth_list = []

            for MODEL in range(len(model_names_str)):
                model_name = model_names_str[MODEL].format(RDIM)
                model_label_start = model_label_str[MODEL]
                model_color = model_color_str[MODEL]
                model_marker = model_marker_str[MODEL]
                model_markeredgewidth = model_marker_width_str[MODEL]

                model_found = False
                REGEX = re.compile(model_name)
                for model_name_key in model_test_dict:
                    model_train = model_train_dict[model_name_key]
                    if REGEX.match(model_name_key):

                        error_metric = model_test_dict[model_name_key][CRITERION]
                        if CRITERION=="num_accurate_pred_050_avg_TEST":
                            error_metric = (-error_metric)


                        if (not model_found) or (error_metric < error_metric_):
                            model_found = True
                            error_metric_ = error_metric
                            val_file_ = val_results_path + model_name_key + "/results.pickle"
                            train_file_ = train_results_path + model_name_key + "/data.pickle"
                            temp = model_name_key.split("_")
                            if temp[0]=='GPU-RNN-esn(.*)-RDIM':
                                SIZE = temp[4].split("-")[0]
                            else:
                                SIZE = temp[5].split("-")[0]
                            model_label=model_label_start+"-{:s}".format(SIZE)
                print(model_label)
                print("SIZE {:}".format(SIZE))

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

                rmnse_all_list.append(rmnse_all)
                prediction_all_list.append(predictions_all)
                truth_all_list.append(truths_all)
                model_color_list.append(model_color)
                model_label_list.append(model_label)
                model_marker_list.append(model_marker)
                markersize_list.append(markersize)
                linewidth_list.append(linewidth)
                model_markeredgewidth_list.append(model_markeredgewidth)

            for IC in [10,15,20,25,30,35,40,45,50,55,60]:
                rmnse_list = []
                error_list = []
                prediction_list = []
                truth_list = []
                time_vector_list = []
                for P in range(len(model_label_list)):
                    prediction = prediction_all_list[P][IC]
                    prediction = replaceInvalid(prediction)
                    target = truth_all_list[P][IC]
                    rmnse = rmnse_all_list[P][IC]
                    error = np.sqrt(pow(prediction-target, 2.0)/pow(scaler.data_std, 2.0))
                    error = replaceInvalid(error)
                    prediction=prediction[:NSTEPS]
                    target=target[:NSTEPS]
                    rmnse=rmnse[:NSTEPS]
                    error=error[:NSTEPS]
                    time_vector = np.arange(np.shape(error)[0])
                    rmnse_list.append(rmnse)
                    prediction_list.append(prediction)
                    truth_list.append(target)
                    error_list.append(error)
                    time_vector_list.append(time_vector)


                NUM_PLOTS=len(model_names_str)+1

                fig, axes = plt.subplots(nrows=2, ncols=NUM_PLOTS,figsize=(20, 12), gridspec_kw={"width_ratios":NUM_PLOTS*[1]}, sharey=True)
                fig.subplots_adjust(hspace=0.2, wspace = 0.4)

                time_min=time_vector.min()*dt*lambda1
                time_max=time_vector.max()*dt*lambda1
                vmin = np.array(target).min()
                vmax = np.array(target).max()
                vmax_abs=np.array([np.abs(vmin), np.abs(vmax)]).max()
                vmin=-vmax_abs
                vmax=+vmax_abs
                vmin_error = 0.0
                vmax_error = 2.0
                # print("VMIN: {:} \n VMAX {:} \n".format(vmin, vmax))

                createContour_(fig, axes[0, 0], target, "TARGET", vmin, vmax, plt.get_cmap("seismic"), dt*lambda1)
                axes[0, 0].set_ylim([time_min, time_max])
                # axes[0, 0].set_xlabel(r"Gridpoint")
                axes[0, 0].set_ylabel(r"$t \, / \, T^{\Lambda_1}$")
                axes[0, 0].set_yticks(np.linspace(0, T_max, T_max+1))


                for P in range(len(model_label_list)):
                    num_samples=np.sum(time_vector_list[P]*dt*lambda1<time_max)
                    markevery=int(num_samples/12)
                    markevery=getPrimeNumbers(len(model_names_str), markevery)[P]
                    axes[1, 0].plot(rmnse_list[P], time_vector_list[P]*dt*lambda1, color=model_color_list[P], label=model_label_list[P], marker=model_marker_list[P], markersize=markersize_list[P], linewidth=linewidth_list[P], markeredgewidth=model_markeredgewidth_list[P],markevery=markevery)

                axes[1, 0].set_title(r"NRMSE")
                axes[1, 0].set_ylabel(r"$t \, / \, T^{\Lambda_1}$")
                axes[1, 0].set_xlabel(r"NRMSE")
                # axes[1, 0].set_xlim([2, -8])
                axes[1, 0].set_ylim([time_min, time_max])
                axes[1, 0].set_yticks(np.linspace(0, T_max, T_max+1))
                axes[1, 0].set_xlim([vmax_error, vmin_error])


                for i in range(len(model_label_list)):
                    prediction = prediction_list[i]
                    truth = truth_list[i]
                    rmnse = rmnse_list[i]
                    error = error_list[i]
                    model_label=model_label_list[i]
                    print("MUST BE ZERO {:}".format(np.linalg.norm(truth-target)))
                    mp = createContour_(fig, axes[0, i+1], prediction, model_label, vmin, vmax, plt.get_cmap("seismic"), dt*lambda1)
                    mp_error = createContour_(fig, axes[1, i+1], error, model_label + " - NRSE", vmin_error, vmax_error, plt.get_cmap("Reds"), dt*lambda1)
                    axes[0, i+1].set_ylim([time_min, time_max])
                    axes[1, i+1].set_ylim([time_min, time_max])
                    # axes[0, i+1].set_xlabel(r"Gridpoint"))
                    axes[1, i+1].set_xlabel(r"Observable")

                cbar_ax = fig.add_axes([0.93, 0.53, 0.015, 0.35]) #[left, bottom, width, height] 
                ticks_=list(np.linspace(vmin,0,4))[:-1]+list(np.linspace(0,vmax, 4))
                cbar=fig.colorbar(mp, cax=cbar_ax, format='%.0f', ticks=ticks_)

                cbar_ax = fig.add_axes([0.93, 0.11, 0.015, 0.35]) #[left, bottom, width, height] 
                ticks_=list(np.linspace(vmin_error,vmax_error,4))
                cbar=fig.colorbar(mp_error, cax=cbar_ax, format='%.1f', ticks=ticks_)


                # plt.show()
                # print(ark)
                CRITERION_SHORT = "NAP" if CRITERION =="num_accurate_pred_050_avg_TEST" else "FREQERROR"
                fig.savefig(fig_path + '/CONTOUR_BBO_{:}_2_M_RDIM_{:}_IC{:d}.png'.format(CRITERION_SHORT, RDIM, IC), bbox_inches='tight', dpi=300)
                
                lgd = ax.legend(loc="upper left", bbox_to_anchor=(1,1))
                fig.savefig(fig_path + '/CONTOUR_BBO_{:}_2_M_RDIM_{:}_IC{:d}_LEGEND.png'.format(CRITERION_SHORT, RDIM, IC), bbox_inches='tight', dpi=300)






