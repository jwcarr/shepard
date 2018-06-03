'''
This is used in conjunction with mf_collect.py to perform the model
fit. collect.py generates candidate weight and noise parameter
settings, and then runs 168 instances of sample.py (on a cluster),
each of which measures the likelihood of a particular participant's
data_out given the language inferred by a model agent given the
participant's data_in (using the candidate parameter settings). This
sample likelihood is then written to a file and processed by
collect.py on the next optimizer iteration.
'''

import numpy as np
import argparse
import os
import model

shape = (8,8)
maxcats = 4
exposures = 4
mcmc_iterations = 5000

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('path', action='store', type=str, help='path to where all the data is')
	parser.add_argument('prior', action='store', type=str, help='type of prior to use (\'simplicity\')')
	parser.add_argument('iteration', action='store', type=int, help='iteration number')
	parser.add_argument('participant_i', action='store', type=int, help='starting participant number to analyze')
	parser.add_argument('participants_per_job', action='store', type=int, help='number of participants per job')
	parser.add_argument('weight', action='store', type=float, help='weighting of the prior (1)')
	parser.add_argument('noise', action='store', type=float, help='probability of noise on production (0.01)')
	args = parser.parse_args()

	for participant_i in range(args.participant_i, args.participant_i + args.participants_per_job):

		# Get participant data_in and data_out
		with open(os.path.join(args.path, 'data', str(participant_i)), 'r') as file:
			data_in, data_out, participant_id, initial_hypothesis = file.read().strip().split('\n')

		# If an initial hypothesis has been set and you're using the
		# simplicity prior, construct the initial hypothesis array.
		# This is a hack to get MH to converge faster in cases where
		# we expect the agent will produce large categories (which can
		# be very time consuming to rectangularize).
		if args.prior == 'simplicity':
			initial_hypothesis = eval(initial_hypothesis)
		else:
			initial_hypothesis = None
		if initial_hypothesis is not None:
			initial_hypothesis = np.full(shape, initial_hypothesis, dtype=int)

		# Get model agent to infer a language based on participant's
		# data_in and then measure likelihood of participant's data_out
		# given agent's inferred language.
		agent = model.Agent(shape, maxcats, args.prior, args.weight, args.noise, exposures, mcmc_iterations)
		agent.learn(eval(data_in), initial_hypothesis)
		lhood = agent._likelihood(eval(data_out), agent.language)

		# Write out weight, noise, measured likelihood, and inferred
		# language to file for collection by collect.py
		with open(os.path.join(args.path, 'cand', str(args.iteration), str(participant_i)), 'w') as file:
			file.write('\n'.join(map(str, ([args.weight, args.noise, lhood, agent.language.flatten().tolist()]))))
