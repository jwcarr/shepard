'''
Module to convert raw model results fresh off the cluster into single
JSON files. Produces similar output to exp_process.py
'''

import json
import numpy as np
import tools
import os

def check_parameters(dict1, dict2):
	for key, value in dict1.items():
		if dict2[key] != value:
			return True
	return False

def process(input_dir, output_file, min_generations):
	results_dict = {'prior':None, 'weight':None, 'noise':None, 'bottleneck':None, 'exposures':None, 'shape':None, 'mincats':None, 'maxcats':None, 'mcmc_samples':None, 'mcmc_iterations':None, 'chains':[]}
	for file_path, file_name in tools.iter_directory(input_dir):
		chain = {'chain_id':file_name, 'generations':[]}
		with open(file_path, 'r') as file:
			for gen_i, line in enumerate(file):
				generation_data = eval(line)
				if generation_data['filtered_agent']:
					continue
				if gen_i == 0:
					parameters = generation_data['model_parameters']
				elif check_parameters(generation_data['model_parameters'], parameters):
					raise ValueError('Mismatch in file %s on line %i' % (file_path, gen_i+1))
				chain['generations'].append(generation_data)
		if len(chain['generations']) > min_generations:
			results_dict['chains'].append(chain)
		else:
			print('Skipping', file_path)
	for key, value in parameters.items():
		results_dict[key] = value
	with open(output_file, 'w') as file:
		file.write(json.dumps(results_dict))


# CONVERT BASIC MODEL RESULTS

for noise in ['0.01', '0.05', '0.1']:
	for bottleneck in ['1', '2', '3', '4']:
		for exposures in ['1', '2', '3', '4']:

			# input_dir = '../data/model_raw/s_1.0_%s_%s_%s/' % (noise, bottleneck, exposures)
			# output_file = '../data/model_sim/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures)
			# process(input_dir, output_file, 50)

			input_dir = '../data/model_raw/i_1.0_%s_%s_%s/' % (noise, bottleneck, exposures)
			output_file = '../data/model_inf/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures)
			process(input_dir, output_file, 50)

			input_dir = '../data/model_raw/i_500.0_%s_%s_%s/' % (noise, bottleneck, exposures)
			output_file = '../data/model_inf/500.0_%s_%s_%s.json' % (noise, bottleneck, exposures)
			process(input_dir, output_file, 50)

# CONVERT MODEL FIT RESULTS

# process('../data/model_raw/s_1.36_0.23_2_4/', '../data/model_sim/1.36_0.23_2_4.json', 50)
# process('../data/model_raw/i_243.3_0.37_2_4/', '../data/model_inf/243.3_0.37_2_4.json', 50)
