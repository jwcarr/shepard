#!/usr/bin/env python

import random
from pymongo import MongoClient

mongo = MongoClient('mongodb://localhost:27018')
db = mongo.shepard

grid = [[0,1,8,9], [2,3,10,11], [4,5,12,13], [6,7,14,15], [16,17,24,25], [18,19,26,27], [20,21,28,29], [22,23,30,31], [32,33,40,41], [34,35,42,43], [36,37,44,45], [38,39,46,47], [48,49,56,57], [50,51,58,59], [52,53,60,61], [54,55,62,63]]

def create_chains(job_id, n_chains):
	for chain_id in range(n_chains):
		create_chain(job_id, chain_id)

def create_chain(job_id, chain_id):
	chain = {
		'job_id' : job_id,
		'chain_id' : chain_id,
		'running_user' : None,
		'finished_users' : [],
		'terminated_users' : [],
		'rejected_users' : [],
		'generations' : [create_generation_zero()]
	}
	db.chains.save(chain)

def create_generation_zero():
	generation = {
		'user_id' : None,
		'partition' : create_random_partition(),
		'training_out' : select_training_material()
	}
	return generation

def create_random_partition():
	partition = [0]*16 + [1]*16 + [2]*16 + [3]*16
	random.shuffle(partition)
	return partition

def select_training_material():
	training_material = []
	for quad in grid:
		random.shuffle(quad)
		training_material.extend(quad[:2])
	return training_material

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser(description='Initialize iterated learning chains')
	parser.add_argument('--new',    '-n', action="store",     type=str, default=None)
	parser.add_argument('--chains', '-c', action="store",     type=int, default=None)
	parser.add_argument('--wipe',   '-w', action="store_true",          default=None)
	args = parser.parse_args()

	if args.wipe:
		db.chains.remove()
	if args.new:
		create_chains(args.new, args.chains)
