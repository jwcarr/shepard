'''
Finds all the unique rectilinear polygons in the iterated learning
experiment data and writes them out to files to be dissected on a
cluster.
'''

import numpy as np
from scipy import ndimage
import json
from os import path

variant_hashes = {}
canonical_hashes = []

def read_json_file(file_path):
	with open(file_path, mode='r', encoding='utf-8') as file:
		data = [json.loads(line) for line in file if len(line) > 1]
	return data

def create_rectlang_file(chunk_array, chunk_hash, chunk_size):
	filename = '../data/rectlang/input/' + str(chunk_size) + '_' + chunk_hash + '_in'
	rects = ['%i %i %i %i %i %i %i %i'%(x, y, x+1, y, x, y+1, x+1, y+1) for (y, x), value in np.ndenumerate(chunk_array) if value == True]
	with open(filename, mode='w', encoding='ASCII') as file:
		file.write('\n'.join(rects))

def clip(arr):
	xmin, xmax = np.where(np.any(arr, axis=0))[0][[0, -1]]
	ymin, ymax = np.where(np.any(arr, axis=1))[0][[0, -1]]
	shape = ((xmax+1)-xmin, (ymax+1)-ymin)
	return arr[ymin:ymax+1, xmin:xmax+1], shape

def hash_chunk(chunk):
	return '-'.join([''.join(['1' if cell == True else '0' for cell in row]) for row in chunk])

def get_variant_hashes(chunk, reverso):
	hashes = {
		hash_chunk(chunk) : reverso.tolist(),
		hash_chunk(np.fliplr(chunk)) : np.fliplr(reverso).tolist(),
		hash_chunk(np.flipud(chunk)) : np.flipud(reverso).tolist(),
		hash_chunk(np.fliplr(np.flipud(chunk))) : np.fliplr(np.flipud(reverso)).tolist(),

		hash_chunk(np.rot90(chunk, 1)) : np.rot90(reverso, 1).tolist(),
		hash_chunk(np.fliplr(np.rot90(chunk, 1))) : np.fliplr(np.rot90(reverso, 1)).tolist(),
		hash_chunk(np.flipud(np.rot90(chunk, 1))) : np.flipud(np.rot90(reverso, 1)).tolist(),
		hash_chunk(np.fliplr(np.flipud(np.rot90(chunk, 1)))) : np.fliplr(np.flipud(np.rot90(reverso, 1))).tolist(),

		hash_chunk(np.rot90(chunk, 2)) : np.rot90(reverso, 2).tolist(),
		hash_chunk(np.fliplr(np.rot90(chunk, 2))) : np.fliplr(np.rot90(reverso, 2)).tolist(),
		hash_chunk(np.flipud(np.rot90(chunk, 2))) : np.flipud(np.rot90(reverso, 2)).tolist(),
		hash_chunk(np.fliplr(np.flipud(np.rot90(chunk, 2)))) : np.fliplr(np.flipud(np.rot90(reverso, 2))).tolist(),

		hash_chunk(np.rot90(chunk, 3)) : np.rot90(reverso, 3).tolist(),
		hash_chunk(np.fliplr(np.rot90(chunk, 3))) : np.fliplr(np.rot90(reverso, 3)).tolist(),
		hash_chunk(np.flipud(np.rot90(chunk, 3))) : np.flipud(np.rot90(reverso, 3)).tolist(),
		hash_chunk(np.fliplr(np.flipud(np.rot90(chunk, 3)))) : np.fliplr(np.flipud(np.rot90(reverso, 3))).tolist(),
	}
	return hashes

def create_reverse_chunk(chunk_hash, chunk_size):
	arr = np.zeros((8,8), dtype=int)
	ex_filename = '../data/rectlang/output_exhaustive/%i_%s_out'%(chunk_size, chunk_hash)
	bm_filename = '../data/rectlang/output_beam/%i_%s_out'%(chunk_size, chunk_hash)
	if path.isfile(ex_filename):
		filename = ex_filename
	elif path.isfile(bm_filename):
		filename = bm_filename
	else:
		return False
	with open(filename, mode='r', encoding='ASCII') as file:
		counter = 1
		for line in file:
			rect = line.split(' ')
			if len(rect) == 8:
				rect = list(map(int, rect))
				arr[rect[1]:rect[7], rect[0]:rect[6]] = counter
				counter += 1
	return clip(arr)[0]

regular_count = {}
irregular_count = { 29:{}, 39:{}, 49:{}, 59:{}, 64:{} }

def evaluate_partition(partition):
	global canonical_hashes, variant_hashes, regular_count, irregular_count
	for category_label in np.unique(partition):
		category = partition == category_label
		chunk_areas, n_chunks = ndimage.label(category)
		for chunk_label in range(1, n_chunks+1):
			chunk, shape = clip(chunk_areas == chunk_label)
			if chunk.all() == False:
				chunk_size = chunk.sum()
				this_hash = hash_chunk(chunk)
				# if this_hash not in variant_hashes:
					# create_rectlang_file(chunk, this_hash, chunk_size)
					# variants = get_variant_hashes(chunk, chunk)
					# reverso = create_reverse_chunk(this_hash, chunk_size)
					# if type(reverso) == bool and reverso == False:
					# 	continue
					# variants = get_variant_hashes(chunk, reverso)
					# for variant_hash in variants:
						# variant_hashes[variant_hash] = variants[variant_hash]
				if this_hash not in variant_hashes:
					variants = get_variant_hashes(chunk, chunk)
					for variant_hash in variants.keys():
						variant_hashes[variant_hash] = this_hash
						canonical_hash = this_hash
				else:
					canonical_hash = variant_hashes[this_hash]

				for size in [29, 39, 49, 59, 64]:
					if chunk_size <= size:
						break
				if canonical_hash in irregular_count[size]:
					irregular_count[size][canonical_hash] += chunk_size
				else:
					irregular_count[size][canonical_hash] = chunk_size
			else:
				shape = tuple(list(sorted(shape)))
				if shape in regular_count:
					regular_count[shape] += 1
				else:
					regular_count[shape] = 1

######################################################################

# data = read_json_file('../data/experiments/exp2_chains.json')
# for chain in range(0,12):
# 	gens = data[chain]['generations']
# 	for gen in range(len(gens)):
# 		partition = np.array(gens[gen]['partition'], dtype=int).reshape((8, 8))
# 		evaluate_partition(partition)

# print(sum(irregular_count[29].values()))
# print(sum(irregular_count[39].values()))
# print(sum(irregular_count[49].values()))
# print(sum(irregular_count[59].values()))
# print(sum(irregular_count[64].values()))
