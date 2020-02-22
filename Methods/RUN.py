#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import sys
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from plotting_utils import *
from global_utils import *

import argparse


def getModel(params):
	sys.path.insert(0, global_params.py_models_path.format(params["model_name"]))
	if params["model_name"] == "esn":
		import esn as model
		return model.esn(params)
	elif params["model_name"] == "esn_parallel":
		import esn_parallel as model
		return model.esn_parallel(params)
	elif params["model_name"] == "rnn_statefull":
		import rnn_statefull as model
		return model.rnn_statefull(params)
	elif params["model_name"] == "rnn_statefull_parallel":
		import rnn_statefull_parallel as model
		return model.rnn_statefull_parallel(params)
	else:
		raise ValueError("model not found.")

def runModel(params_dict):
	if params_dict["mode"] in ["train", "all"]:
		trainModel(params_dict)
	if params_dict["mode"] in ["test", "all"]:
		testModel(params_dict)
	return 0

def trainModel(params_dict):
	model = getModel(params_dict)
	model.train()
	model.delete()
	del model
	return 0

def testModel(params_dict):
	model = getModel(params_dict)
	model.testing()
	model.delete()
	del model
	return 0


def defineParser():
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(help='Selection of the model.', dest='model_name')

	esn_parser = subparsers.add_parser("esn")
	esn_parser = getESNParser(esn_parser)
	
	esn_parallel_parser = subparsers.add_parser("esn_parallel")
	esn_parallel_parser = getESNParallelParser(esn_parallel_parser)

	rnn_statefull_parser = subparsers.add_parser("rnn_statefull")
	rnn_statefull_parser = getRNNStatefullParser(rnn_statefull_parser)
	
	rnn_statefull_parallel_parser = subparsers.add_parser("rnn_statefull_parallel")
	rnn_statefull_parallel_parser = getRNNStatefullParallelParser(rnn_statefull_parallel_parser)
	
	mlp_parser = subparsers.add_parser("mlp")
	mlp_parser = getMLPParser(mlp_parser)
	
	mlp_parallel_parser = subparsers.add_parser("mlp_parallel")
	mlp_parallel_parser = getMLPParallelParser(mlp_parallel_parser)
	
	return parser

def main():
	parser = defineParser()
	args = parser.parse_args()
	print(args.model_name)
	args_dict = args.__dict__

	# for key in args_dict:
		# print(key)

	# DEFINE PATHS AND DIRECTORIES
	args_dict["saving_path"] = global_params.saving_path.format(args_dict["system_name"])
	args_dict["model_dir"] = global_params.model_dir
	args_dict["fig_dir"] = global_params.fig_dir
	args_dict["results_dir"] = global_params.results_dir
	args_dict["logfile_dir"] = global_params.logfile_dir
	args_dict["train_data_path"] = global_params.training_data_path.format(args.system_name, args.N)
	args_dict["test_data_path"] = global_params.testing_data_path.format(args.system_name, args.N)
	args_dict["worker_id"] = 0

	runModel(args_dict)

if __name__ == '__main__':
	main()

