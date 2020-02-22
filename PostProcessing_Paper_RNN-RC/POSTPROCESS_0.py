#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import pickle
import glob, os
import numpy as np
import argparse
from Utils.utils import *

# ADDING PARENT DIRECTORY TO PATH
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
methods_dir = os.path.dirname(current_dir)+"/Methods"
sys.path.insert(0, methods_dir) 
from Config.global_conf import global_params
global_utils_path = methods_dir + "/Models/Utils"
sys.path.insert(0, global_utils_path) 
from global_utils import *



parser = argparse.ArgumentParser()
parser.add_argument("--system_name", help="system", type=str, required=True)
parser.add_argument("--Experiment_Name", help="Experiment_Name", type=str, required=False, default=None)
args = parser.parse_args()
system_name = args.system_name
Experiment_Name = args.Experiment_Name

# system_name="Lorenz3D"
# Experiment_Name="Experiment_Daint_Large"
# python3 POSTPROCESS_0.py --system_name Lorenz96_F8GP40R40 --Experiment_Name="Experiment_Daint_Large"

if Experiment_Name is None or Experiment_Name=="None" or global_params.cluster == "daint":
    saving_path = global_params.saving_path.format(system_name)
else:
    saving_path = global_params.saving_path.format(Experiment_Name +"/"+system_name)
logfile_path=saving_path+"/Logfiles"
print(system_name)
print(logfile_path)

modellist = getAllModelsTestList(logfile_path)

COLUMN_TO_SORT=2
COLUMN_NAME="TEST_050"
modellist_sorted = sortModelList(modellist, COLUMN_TO_SORT)

filename='./test_sorted_{:}.txt'.format(COLUMN_NAME)
with open(filename, 'w') as file_object:
    header_line = "#RANK# TEST_005 # TEST_050 # TRAIN_005# TRAIN_050# FE_TRAIN #  FE_TEST # NAME \n"
    file_object.write(header_line)
    iter_=0
    while(len(modellist_sorted)>iter_):
        model = modellist_sorted[iter_]
        line="#{:^4s}".format(str(iter_))
        for element in model[1:]:
            # print(element)
            # print(line)
            line+="#{:^10s}".format(str(element))
        line+= "# {:} ".format(model[0])
        line+="\n"
        # print(line)
        file_object.write(line)
        iter_+=1

COLUMN_TO_SORT=4
COLUMN_NAME="TRAIN_050"
modellist_sorted = sortModelList(modellist, COLUMN_TO_SORT)

filename='./test_sorted_{:}.txt'.format(COLUMN_NAME)
with open(filename, 'w') as file_object:
    header_line = "#RANK# TEST_005 # TEST_050 # TRAIN_005# TRAIN_050# FE_TRAIN #  FE_TEST # NAME \n"
    file_object.write(header_line)
    iter_=0
    while(len(modellist_sorted)>iter_):
        model = modellist_sorted[iter_]
        line="#{:^4s}".format(str(iter_))
        for element in model[1:]:
            # print(element)
            # print(line)
            line+="#{:^10s}".format(str(element))
        line+= "# {:} ".format(model[0])
        line+="\n"
        # print(line)
        file_object.write(line)
        iter_+=1






