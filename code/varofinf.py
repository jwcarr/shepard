'''
Computes the variation of information, an information theoretic
measure of the distance between two set partitions. Used as the
measure of transmission error.
'''

import numpy as np

def language_to_partition(language):
	'''
	Converts a language array to a list of category sets.
	'''
	category_sets = {}
	for meaning, signal in enumerate(language.flatten()):
		if signal in category_sets:
			category_sets[signal].add(meaning)
		else:
			category_sets[signal] = set([meaning])
	return list(category_sets.values())

def variation_of_information(language1, language2):
	'''
	Computes the variation of information, an information theoretic
	measure of the distance between two set partitions. Used as the
	measure of transmission error.
	'''
	if language1.shape != language2.shape:
		raise ValueError('The shapes of the partitions do not match.')
	n = float(language1.size)
	partition1 = language_to_partition(language1)
	partition2 = language_to_partition(language2)
	sigma = 0.0
	for category1 in partition1:
		p = len(category1) / n
		for category2 in partition2:
			q = len(category2) / n
			r = len(category1 & category2) / n
			if r > 0.0:
				sigma += r * (np.log2(r / p) + np.log2(r / q))
	return abs(sigma)
