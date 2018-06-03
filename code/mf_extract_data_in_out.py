'''
Get data_in and data_out for all participants whose VI is less than
3.0 and write them out to files for model fitting.
'''

import numpy as np
import tools

initial_hypotheses = {'C2':1, 'C3':1, 'C4':1, 'C5':1, 'C6':1, 'C7':1, 'C8':1, 'C9':1, 'C10':1, 'D8':2, 'D9':2, 'D10':2, 'D11':2, 'D12':2, 'D13':2, 'F13':2, 'F14':2, 'F15':2, 'F16':2, 'G8':2, 'G9':2, 'G10':2, 'G11':2, 'J13':2, 'J14':2, 'J15':2, 'J16':2, 'J17':2, 'J18':2, 'J19':2, 'J20':2, 'J21':2, 'J22':2, 'K17':3, 'K18':3, 'K19':3, 'L26':2, 'L27':2, 'L28':2, 'L29':2, 'L30':2}

results = tools.read_json_file('../data/experiments/exp2_chains.json')

chain_letters = 'ABCDEFGHIJKL'
participant_i = 1

for c, chain in enumerate(results['chains']):
	for g, generation in enumerate(chain['generations']):
		error = generation['prod_error']
		if error is not None and error < 3.0: # exclude participants whose VI is greater than 3.0
			data_in = [tuple([tuple(meaning), signal]) for meaning, signal in chain['generations'][g-1]['data_out']]
			data_out = [tuple([tuple(meaning), signal]) for meaning, signal in np.ndenumerate(np.array(generation['productions'], dtype=int).reshape((8,8)))]
			participant_id = chain_letters[c] + str(g)
			if participant_id in initial_hypotheses:
				initial_hypothesis = initial_hypotheses[participant_id]
			else:
				initial_hypothesis = None
			with open('../data/model_fit/dat/%i' % participant_i, mode='w') as file:
				file.write('\n'.join([str(data_in), str(data_out), participant_id, str(initial_hypothesis)]))
			participant_i += 1

print(participant_i)
