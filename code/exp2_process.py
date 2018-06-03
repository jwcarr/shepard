'''
Converts the raw results from chains.json into the same JSON format
used by the model. Produces similar output to model_process.py
'''

import json
import numpy as np
import rectlang
import commcost
import varofinf
import tools

shape = (8,8)

rectlang_space = rectlang.Space(shape, solutions_file='../data/8x8_solutions.json')
commcost_space = commcost.Space(shape)

def raw_model_results_to_json_files(input_file, output_file):
	results = {'bottleneck':2, 'exposures':4, 'chains':[]}
	data = tools.read_json_lines(input_file)
	for chain in data:
		chain_data = {'chain_id':chain['chain_id'], 'first_fixation':chain['first_fixation'], 'generations':[]}
		for gen_i, generation in enumerate(chain['generations']):
			generation_data = {'generation_number':gen_i, 'productions':generation['partition'], 'data_out':[]}
			productions = np.array(generation['partition'], dtype=int).reshape((8,8))
			for stim_i in generation['training_out']:
				meaning = (stim_i // shape[0], stim_i % shape[1])
				signal = int(productions[meaning])
				generation_data['data_out'].append((meaning, signal))
			generation_data['prod_expressivity'] = len(np.unique(productions))
			generation_data['prod_cost'] = commcost_space.cost(productions)
			generation_data['prod_complexity'] = rectlang_space.complexity(productions)
			if gen_i > 0:
				generation_data['prod_error'] = varofinf.variation_of_information(prev_productions, productions)
			else:
				generation_data['prod_error'] = None
			prev_productions = productions
			chain_data['generations'].append(generation_data)
		results['chains'].append(chain_data)
	with open(output_file, 'w') as file:
		file.write(json.dumps(results))

######################################################################

# raw_model_results_to_json_files('../data/experiments_raw/chains.json', '../data/experiments/exp2_chains.json')
