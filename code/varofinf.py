'''
Computes the variation of information, an information theoretic
measure of the distance between two set partitions. Used as the
measure of transmission error.
'''

import numpy as np

def language_to_category_sets(partiton):
	'''
	Converts a partiton array to a list of category sets.
	'''
	partiton = partiton.flatten()
	category_sets = {}
	for i in range(len(partiton)):
		if partiton[i] in category_sets.keys():
			category_sets[partiton[i]].append(i)
		else:
			category_sets[partiton[i]] = [i]
	return [vals for vals in category_sets.values()]

def variation_of_information(partition1, partition2):
	'''
	Computes the variation of information, an information theoretic
	measure of the distance between two set partitions. Used as the
	measure of transmission error.
	'''
	if partition1.shape != partition2.shape:
		raise ValueError('The shapes of the partitions do not match.')
	n = partition1.size
	partition1 = language_to_category_sets(partition1)
	partition2 = language_to_category_sets(partition2)
	sigma = 0.0
	for cat1 in partition1:
		p = len(cat1) / n
		for cat2 in partition2:
			q = len(cat2) / n
			r = len(set(cat1) & set(cat2)) / n
			if r > 0.0:
				sigma += r * (np.log2(r / p) + np.log2(r / q))
	return abs(-sigma)
