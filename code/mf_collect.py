'''
This is used in conjunction with mf_sample.py to perform the model
fit. collect.py generates candidate weight and noise parameter
settings, and then runs 168 instances of sample.py (on a cluster),
each of which measures the likelihood of a particular participant's
data_out given the language inferred by a model agent given the
participant's data_in (using the candidate parameter settings). This
sample likelihood is then written to a file and processed by
collect.py on the next optimizer iteration.
'''

from subprocess import call, check_output
import numpy as np
import argparse
import pickle
import os
import re

n_participants = 168
n_random_points = 200
n_iterations = 400
shape = (8,8)
maxcats = 4

# Optimization bounds for each prior:

bounds      = {'simplicity':      [(0.0, 4.0),    (0.0001, 0.9999)],
               'informativeness': [(0.0, 1000.0), (0.0001, 0.9999)]}

# Requested cluster resources for each prior:

resources   = {'simplicity':      {'mins':'08', 'memory':'700M', 'participants_per_job':1},
               'informativeness': {'mins':'04', 'memory':'600M', 'participants_per_job':8}}

# Shell script for submitting jobs to the cluster:

template_script = '''#!/bin/sh
#$ -N {job_name}
#$ -cwd
#$ -l h_rt=00:{mins}:00
#$ -pe sharedmem 1
#$ -l h_vmem={memory}
#$ -o ~/scratch/logs/
#$ -e ~/scratch/logs/

ulimit -c 0
export OMP_NUM_THREADS=1

. /etc/profile.d/modules.sh

module load anaconda
source activate modpy

python mf_sample.py {path} {prior} {iteration} $SGE_TASK_ID {participants_per_job} {weight} {noise}
'''

def restore_optimizer(path, iteration):
	'''
	Restores the state of the optimizer at a given iteration number.
	'''
	with open(os.path.join(path, 'opt', str(iteration)), mode='rb') as file:
		opt = pickle.load(file)
	return opt

def pickle_optimizer(path, iteration, opt, res):
	'''
	Pickles the state of the optimizer and stores to a file based on
	the iteration number.
	'''
	opt.models = opt.models[-1:] # just keep the most recent model
	with open(os.path.join(path, 'opt', str(iteration)), mode='wb') as file:
		pickle.dump(opt, file)
	with open(os.path.join(path, 'result'), mode='wb') as file:
		pickle.dump(res, file)

def collect_results(path, kind, iteration):
	'''
	Collect the likelihoods for each of the participants, and take
	the product to get the probability of the dataset as a whole
	under certain parameter settings.
	'''
	likelihoods = np.zeros(n_participants, dtype=float)
	prev_weight, prev_noise = None, None
	for participant_i in range(1, n_participants+1):
		with open(os.path.join(path, kind, str(iteration), str(participant_i))) as file:
			weight, noise, lhood, language = file.read().strip().split('\n')
		if prev_weight and (weight != prev_weight or noise != prev_noise):
			raise ValueError('Participant %i has weight or noise mismatch' % participant_i)
		prev_weight, prev_noise = weight, noise
		likelihoods[participant_i-1] = float(lhood)

	p_dataset = likelihoods.sum() # sum is product in log domain

	return float(weight), float(noise), -p_dataset # negative because optimizer is minimizing

def create_job_script(path, prior, iteration, weight, noise):
	'''
	Create a new job script for submission to the scheduler.
	'''
	job_name = 'opt-%i' % iteration
	script = template_script.format(job_name=job_name, path=path, prior=prior, iteration=iteration, weight=weight, noise=noise, **resources[prior])
	with open('opt.sh', mode='w') as file:
		file.write(script)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('path', action='store', type=str, help='path where everything is')
	parser.add_argument('prior', action='store', type=str, help='type of prior to use (\'simplicity\')')
	parser.add_argument('iteration', action='store', type=int, help='iteration number')
	args = parser.parse_args()

	if args.iteration == 0:

		# First iteration: Create the optimizer, load in initial
		# pre-computed random points, and then ask for the first
		# candidate parameter settings to try.

		from skopt import Optimizer
		opt = Optimizer(bounds[args.prior], 'ET', n_initial_points=n_random_points)
		for rand_i in range(n_random_points):
			weight, noise, neg_p_dataset = collect_results(args.path, 'rand', rand_i)
			res = opt.tell([weight, noise], neg_p_dataset)
		weight, noise = opt.ask() # get candidate parameter settings to try
		pickle_optimizer(args.path, args.iteration, opt, res)

	else:

		# Subsequent iterations: Collect results from the previous
		# iteration, report these to the optimizer, and then ask for
		# new candidate parameter settings to try.

		weight, noise, neg_p_dataset = collect_results(args.path, 'cand', args.iteration-1)
		opt = restore_optimizer(args.path, args.iteration-1)
		res = opt.tell([weight, noise], neg_p_dataset)
		weight, noise = opt.ask() # get candidate parameter settings to try
		pickle_optimizer(args.path, args.iteration, opt, res)
	
	if args.iteration < n_iterations:

		# Unless the target number of iterations has been reached,
		# create a job script and submit it to the scheduler. Then
		# submit finish.sh to the scheduler, which runs when the main
		# array job has completed. finish.sh runs collect.py (via a 
		# crazy mechanism that you really don't want to know about)
		# with the next iteration number to start the process over.

		results_iteration_path = os.path.join(args.path, 'cand', str(args.iteration))
		if os.path.exists(results_iteration_path):
			raise ValueError('Results path already exists: %s' % results_iteration_path)
		os.makedirs(results_iteration_path)

		create_job_script(args.path, args.prior, args.iteration, weight, noise)
		ppj = resources[args.prior]['participants_per_job']
		qsub_out = check_output(['qsub', '-t', '1-%i:%i' % (n_participants, ppj), 'opt.sh'])
		print(str(qsub_out)[2:-3])
		job_id = int(re.search(r'Your job-array (\d+)\.', str(qsub_out)).group(1))
		call(['qsub', '-hold_jid', str(job_id), 'finish.sh', str(args.iteration)])
