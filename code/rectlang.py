'''
Python module for calculating the coding lengths of concepts or
languages (collections of concepts) in a 2-dimensional space,
following a method first suggested by Fass & Feldman (2002; Advances
in Neural Information Processing Systems). Given a 2D boolean array
representing where members of a concept are in a 2D space, the program
compresses the concept by representing it as a small set of
rectangles. Alternatively, a language can be compressed by passing an
int array, where each integer represents one of the concepts in the
language. The codelengths yielded by this program can be used to
estimate the prior probability of a concept or language under a
compression/simplicity based prior. To use, create a Space object:

	space = Space(shape=(8,8))

specifying the shape (height and width) of the space and then pass a
Numpy array (bool or int) to one of the following methods:

	space.complexity(int_array)
		returns shortest codelength (float) of the language

	space.compress_concept(bool_array)
		returns shortest codelength (float) and the set of rectangles
		needed to represent the concept (list of tuples)

	space.compress_language(int_array)
		returns shortest codelength (float) and the set of rectangles
		needed to represent the language (list of lists of tuples)

Methods are also provided to encode and decode concepts or languages
to and from binary strings (using the Shannon-Fano code):

	space.encode_concept(bool_array) -> binary_string
	space.decode_concept(binary_string) -> bool_array
	space.encode_language(int_array) -> list_of_binary_strings
	space.decode_language(list_of_binary_strings) -> int array

and a method is provided to print the code used for a given space:

	space.tabulate(show_symbols=False)

All locations and sizes are represented height/row/y-axis first and
width/column/x-axis second. The compression process (finding the set
of rectangles that minimize codelength) becomes very slow for large
concepts (> 20 concept members), but cacheing, beam search, clipping,
and initial dissection are implemented to mitigate this somewhat. A
faster method, which is very complicated to implement, would involve
iterating over the minimal dissections of a rectilinear polygon to
find the dissection that yields the shortest codelength.
'''

from collections import defaultdict
from functools import lru_cache
from scipy import ndimage
import numpy as np

CACHE_SIZE = 2**23 # 2**23 requires up to 5 - 10 GB RAM

class Space:

	'''
	A Space object is initialized with the following arguments:

	shape (tuple) : The height and width of the space.

	rotational_invariance (bool) : Codelengths are calculated based
	on the number of possible positions a rectangle can take in the
	space. This setting determines whether or not rotations are
	included in that calculation. The original Fass & Feldmen
	method sets this to True.
	
	max_exhaustive_size (int) : The maximum chunk size that will be
	evaluated exhaustively. Larger chunks will be evaluated by beam
	search. The chunk size is the number of rectangles that a chunk
	(contiguous region) is composed of after initial dissection.
	
	max_beam_width (int) : The width of the beam used in beam
	searches. Larger values are more likely to find the optimal
	solution but take longer to run.
	'''

	hole_struct = np.ones((3, 3), dtype=int)

	def __init__(self, shape, rotational_invariance=True, max_exhaustive_size=16, max_beam_width=2, solutions_file=None):
		if not isinstance(shape, tuple) or len(shape) != 2:
			raise ValueError('shape should be a tuple containing the height and width')
		if not isinstance(shape[0], int) or shape[0] < 1:
			raise ValueError('height must be positive integer')
		self.height = shape[0]
		
		if not isinstance(shape[1], int) or shape[1] < 1:
			raise ValueError('width must be postive integer')
		self.width = shape[1]
		self.shape = (self.height, self.width)
		self.size = self.height * self.width
		
		if not isinstance(rotational_invariance, bool):
			raise ValueError('rotational_invariance must be boolean')
		self.rotational_invariance = rotational_invariance
		
		if max_exhaustive_size is not None:
			if not isinstance(max_exhaustive_size, int) or max_exhaustive_size < 0 or max_exhaustive_size > self.size:
				raise ValueError('max_exhaustive_size must be integer between 0 and %i' % self.size)
			self.max_exhaustive_size = max_exhaustive_size
		else:
			self.max_exhaustive_size = self.size
		
		if not isinstance(max_beam_width, int) or max_beam_width < 1:
			raise ValueError('max_beam_width must be integer greater than 0')
		self.max_beam_width = max_beam_width

		if solutions_file is not None:
			with open(solutions_file, mode='r') as file:
				self._precomputed_solutions = eval(file.read())
		else:
			self._precomputed_solutions = None

		self._rectangle_shapes()
		self._rectangle_symbols()
		self._codelength_lookup()
		self._shannon_fano_code()

	##################
	#  INIT METHODS  #
	##################

	def _rectangle_shapes(self):
		'''
		Returns a list of rectangle shapes for the given space.
		'''
		if self.rotational_invariance:
			if self.height > self.width:
				self.rectangle_shapes = [(m, n) for n in range(1, self.width+1) for m in range(n, self.height+1)]
			else:
				self.rectangle_shapes = [(m, n) for m in range(1, self.height+1) for n in range(m, self.width+1)]
		else:
			self.rectangle_shapes = [(m, n) for m in range(1, self.height+1) for n in range(1, self.width+1)]

	def _rectangle_symbols(self):
		'''
		Returns a dictionary mapping shapes to particular rectangle
		symbols.
		'''
		self.rectangle_symbols = {}
		for m, n in self.rectangle_shapes:
			self.rectangle_symbols[m, n] = []
			for y in range(self.height + 1 - m):
				for x in range(self.width + 1 - n):
					self.rectangle_symbols[m, n].append((m, n, y, x))
			if self.rotational_invariance and m != n:
				for y in range(self.height + 1 - n):
					for x in range(self.width + 1 - m):
						self.rectangle_symbols[m, n].append((n, m, y, x))

	def _codelength_lookup(self):
		'''
		Creates a lookup table so that the rectangularize_ methods
		can quickly retrieve the coding length of a particular class.
		'''
		self.codelength_lookup = {}
		for m, n in self.rectangle_shapes:
			codelength = np.log2(len(self.rectangle_shapes) * len(self.rectangle_symbols[m, n]))
			self.codelength_lookup[m, n] = codelength
			if self.rotational_invariance and m != n:
				if m <= self.width and n <= self.height:
					self.codelength_lookup[n, m] = codelength

	def _shannon_fano_code(self):
		'''
		Gets the set of possible symbols in the rectangle language,
		calculates their probabilities, and derives the (non-optimal)
		Shannon-Fano codewords for those symbols.
		'''
		symbol_probabilities = []
		for shape in self.rectangle_shapes:
			probability = 2 ** -self.codelength_lookup[shape]
			symbol_probabilities.extend([(probability, symbol) for symbol in self.rectangle_symbols[shape]])
		symbol_probabilities.sort(reverse=False)
		probabilities, symbols = zip(*symbol_probabilities)
		self.symbol_to_string, self.string_to_symbol = {}, {}
		self._shannon_fano(list(symbols), list(probabilities))

	def _shannon_fano(self, symbols, probabilities, codeword=''):
		'''
		Recursive implementation of the Shannon-Fano algorithm.
		'''
		if len(symbols) == 1:
			self.symbol_to_string[symbols[0]] = codeword
			self.string_to_symbol[codeword] = symbols[0]
			return
		previous_difference = 1.0
		for bifurcation in range(1, len(symbols)):
			difference = abs(sum(probabilities[:bifurcation]) - sum(probabilities[bifurcation:]))
			if difference > previous_difference:
				bifurcation -= 1
				break
			previous_difference = difference
		self._shannon_fano(symbols[:bifurcation], probabilities[:bifurcation], codeword+'0')
		self._shannon_fano(symbols[bifurcation:], probabilities[bifurcation:], codeword+'1')

	##################
	# HELPER METHODS #
	##################

	def _sum_codelength(self, rects):
		'''
		Takes a set of rectangles and returns the sum coding length.
		'''
		codelength = 0.0
		for rect in rects:
			codelength += self.codelength_lookup[rect[4]] # 4th element is the rect's size
		return codelength

	def _check_for_precomputed_solution(self, chunk_array):
		'''
		Checks to see if the solution to a given chunk has already
		been preloaded from a precomputed solutions file. This is
		done by hashing the chunk and looking for a match in the
		Grid._precomputed_solutions dictionary. If a match is found,
		the coding length and rectangles are returned.
		'''
		if self._precomputed_solutions:
			chunk_hash = '-'.join([''.join(['1' if cell == True else '0' for cell in row]) for row in chunk_array])
			if chunk_hash in self._precomputed_solutions:
				solution = np.array(self._precomputed_solutions[chunk_hash], dtype=int)
				codelength = 0.0
				rectangles = []
				for rect_i in range(1, solution.max()+1):
					rmin, rmax, cmin, cmax = self._get_chunk_bounding_box(solution == rect_i)
					rectangle = ((rmin, cmin), (rmax, cmin), (rmin, cmax), (rmax, cmax), (rmax-rmin, cmax-cmin))
					codelength += self.codelength_lookup[rectangle[4]]
					rectangles.append(rectangle)
				return codelength, tuple(rectangles)
		return False

	def _get_chunk_bounding_box(self, chunk_array):
		'''
		Returns the bounding box around a chunk.
		'''
		rmin, rmax = np.where(np.any(chunk_array, axis=1))[0][[0, -1]]
		cmin, cmax = np.where(np.any(chunk_array, axis=0))[0][[0, -1]]
		return rmin, rmax+1, cmin, cmax+1

	def _clip_chunk(self, chunk_array):
		'''
		Clips a chunk to its bounding box, and returns rmin and cmin
		so that the chunk can later be unclipped.
		'''
		rmin, rmax, cmin, cmax = self._get_chunk_bounding_box(chunk_array)
		clipped_chunk_array = chunk_array[rmin:rmax+1, cmin:cmax+1]
		return clipped_chunk_array, rmin, cmin

	def _unclip_chunk(self, rects, rmin, cmin):
		'''
		Given rmin and cmin, move the rectangles back to their
		original locations prior to clipping.
		'''
		chunk_rectangles = []
		for rect in rects:
			rect = ((rect[0][0]+rmin, rect[0][1]+cmin), 
					(rect[1][0]+rmin, rect[1][1]+cmin),
					(rect[2][0]+rmin, rect[2][1]+cmin),
					(rect[3][0]+rmin, rect[3][1]+cmin),
					rect[4])
			chunk_rectangles.append(rect)
		return chunk_rectangles

	def _identify_concave_vertices(self, array, hole=False):
		'''
		Scans an array looking for adjacent cells that are unequal to
		find the set of vertices that circumscribe the chunk. It then
		iterates over these vertices to identify those that are
		concave.
		'''
		height, width = array.shape
		array_no_holes = ndimage.morphology.binary_fill_holes(array, self.hole_struct)
		padded_array = np.zeros((height+2, width+2))
		padded_array[1:height+1, 1:width+1] = array_no_holes
		vertex_map = defaultdict(list)
		for y in range(padded_array.shape[0]-1):
			for x in range(padded_array.shape[1]-1):
				if padded_array[y,x] != padded_array[y+1,x]:
					vertex_map[(y, x-1)].append((y, x))
					vertex_map[(y, x)].append((y, x-1))
				if padded_array[y,x] != padded_array[y,x+1]:
					vertex_map[(y-1, x)].append((y, x))
					vertex_map[(y, x)].append((y-1, x))
		vertices = sorted(vertex_map.keys())
		vertices = [vertices[1], vertices[0]]
		concave_vertices = []
		while vertices[0] != vertices[-1]:
			cur = vertices[-1]
			prv = vertices[-2]
			if cur[1] < prv[1]: #left
				nxt = cur[0]+1, cur[1] #down
				if nxt in vertex_map[cur]:
					if hole:
						concave_vertices.append(vertices[-1])
					vertices.append(nxt)
					continue
				nxt = cur[0], cur[1]-1 #left
				if nxt in vertex_map[cur]:
					vertices.append(nxt)
					continue
				nxt = cur[0]-1, cur[1] #up
				if nxt in vertex_map[cur]:
					if not hole:
						concave_vertices.append(vertices[-1])
					vertices.append(nxt)
					continue
			if cur[0] > prv[0]: #down
				nxt = cur[0], cur[1]+1 #right
				if nxt in vertex_map[cur]:
					if hole:
						concave_vertices.append(vertices[-1])
					vertices.append(nxt)
					continue
				nxt = cur[0]+1, cur[1] #down
				if nxt in vertex_map[cur]:
					vertices.append(nxt)
					continue
				nxt = cur[0], cur[1]-1 #left
				if nxt in vertex_map[cur]:
					if not hole:
						concave_vertices.append(vertices[-1])
					vertices.append(nxt)
					continue
			if cur[1] > prv[1]: #right
				nxt = cur[0]-1, cur[1] #up
				if nxt in vertex_map[cur]:
					if hole:
						concave_vertices.append(vertices[-1])
					vertices.append(nxt)
					continue
				nxt = cur[0], cur[1]+1 #right
				if nxt in vertex_map[cur]:
					vertices.append(nxt)
					continue
				nxt = cur[0]+1, cur[1] #down
				if nxt in vertex_map[cur]:
					if not hole:
						concave_vertices.append(vertices[-1])
					vertices.append(nxt)
					continue
			if cur[0] < prv[0]: #up
				nxt = cur[0], cur[1]-1 #left
				if nxt in vertex_map[cur]:
					if hole:
						concave_vertices.append(vertices[-1])
					vertices.append(nxt)
					continue
				nxt = cur[0]-1, cur[1] #up
				if nxt in vertex_map[cur]:
					vertices.append(nxt)
					continue
				nxt = cur[0], cur[1]+1 #right
				if nxt in vertex_map[cur]:
					if not hole:
						concave_vertices.append(vertices[-1])
					vertices.append(nxt)
					continue
		return concave_vertices

	def _initial_dissection(self, chunk_array):
		'''
		Identifies all concave vertices in a chunk and uses these to
		dissect the array into an initial set of rectangles. In many
		cases, this allows the merge process to begin from a smaller
		set than all the individual 1x1 cells.
		'''
		concave_vertices = self._identify_concave_vertices(chunk_array, hole=False)
		holes, n_holes = ndimage.label(ndimage.morphology.binary_fill_holes(chunk_array, self.hole_struct) ^ chunk_array)
		for hole_label in range(1, n_holes+1):
			hole_array = holes == hole_label
			concave_vertices.extend(self._identify_concave_vertices(hole_array, hole=True))
		h_splits, v_splits = set(), set()
		for vertex in set(concave_vertices):
			h_splits.add(vertex[0])
			v_splits.add(vertex[1])
		h_splits, v_splits = sorted(list(h_splits)), sorted(list(v_splits))
		h_iterator = list(zip([0]+h_splits, h_splits+[chunk_array.shape[0]]))
		v_iterator = list(zip([0]+v_splits, v_splits+[chunk_array.shape[1]]))
		rects = []
		for start_h, end_h in h_iterator:
			for start_v, end_v in v_iterator:
				box = chunk_array[start_h:end_h, start_v:end_v]
				if box.all():
					height, width = box.shape
					rect = ((start_h, start_v), (start_h, start_v+width), (start_h+height, start_v), (start_h+height, start_v+width), (height, width))
					rects.append(rect)
				elif box.any():
					subparts, n_subparts = ndimage.label(box)
					for subpart_label in range(1, n_subparts+1):
						subpart = subparts == subpart_label
						rmin, rmax, cmin, cmax = self._get_chunk_bounding_box(subpart)
						rect = ((start_h+rmin, start_v+cmin), (start_h+rmin, start_v+cmax), (start_h+rmax, start_v+cmin), (start_h+rmax, start_v+cmax), (rmax-rmin, cmax-cmin))
						rects.append(rect)
		return tuple(sorted(rects))

	@lru_cache(maxsize=CACHE_SIZE)
	def _exhaustive_merge(self, rects):
		'''
		Takes a set of rectangles (ordered rightwards and downwards)
		and attempts to merge them together recursively until no more
		mergers are possible. The search space of possible merge
		sequences is explored exhaustively to find the set of
		rectangles with the minimum total coding length.
		'''
		n_rects = len(rects)
		for i in range(n_rects):
			for j in range(i+1, n_rects):
				if rects[i][1] == rects[j][0] and rects[i][3] == rects[j][2]: # horizontal merger
					merged_rect = (rects[i][0], rects[j][1], rects[i][2], rects[j][3], (rects[i][2][0]-rects[i][0][0], rects[j][1][1]-rects[i][0][1]))
				elif rects[i][2] == rects[j][0] and rects[i][3] == rects[j][1]: # vertical merger
					merged_rect = (rects[i][0], rects[i][1], rects[j][2], rects[j][3], (rects[j][2][0]-rects[i][0][0], rects[i][1][1]-rects[i][0][1]))
				else: # no merger is possible, so skip ahead to next j
					continue
				next_rects = rects[:i] + (merged_rect,) + rects[i+1:j] + rects[j+1:] # remove rects i and j and insert the merged rect
				codelength = self._sum_codelength(next_rects)
				if codelength < self._best_merge_codelength:
					self._best_merge_codelength = codelength
					self._best_merge_rectangles = next_rects
				self._exhaustive_merge(next_rects)

	@lru_cache(maxsize=CACHE_SIZE)
	def _beam_merge(self, rects):
		'''
		Beam search version of the exhaustive algorithm above. Rather
		than recurse on every branch, select the n most promising
		branches (based on total coding length) and recurse on those,
		where n is the class-level attribute Space.max_exhaustive_size.
		This is significantly faster for larger chunk sizes, but is
		not guaranteed to yield the best possible rectangularization.
		'''
		beam = []
		n_rects = len(rects)
		for i in range(n_rects):
			for j in range(i+1, n_rects):
				if rects[i][1] == rects[j][0] and rects[i][3] == rects[j][2]: # horizontal merger
					merged_rect = (rects[i][0], rects[j][1], rects[i][2], rects[j][3], (rects[i][2][0]-rects[i][0][0], rects[j][1][1]-rects[i][0][1]))
				elif rects[i][2] == rects[j][0] and rects[i][3] == rects[j][1]: # vertical merger
					merged_rect = (rects[i][0], rects[i][1], rects[j][2], rects[j][3], (rects[j][2][0]-rects[i][0][0], rects[i][1][1]-rects[i][0][1]))
				else: # no merger is possible, so skip ahead to next j
					continue
				next_rects = rects[:i] + (merged_rect,) + rects[i+1:j] + rects[j+1:] # remove rects i and j and insert the merged rect
				codelength = self._sum_codelength(next_rects)
				beam.append((codelength, next_rects))
		if len(beam) > 0:
			beam.sort() # sort the beam by coding length
			beam = beam[:self.max_beam_width] # take the most promising branches
			best_beam_codelength, best_beam_rectangles = beam[0]
			if best_beam_codelength < self._best_merge_codelength:
				self._best_merge_codelength = best_beam_codelength
				self._best_merge_rectangles = best_beam_rectangles
			for codelength, next_rects in beam:
				self._beam_merge(next_rects)

	@lru_cache(maxsize=CACHE_SIZE)
	def _rectangularize(self, chunk_array):
		'''
		Takes a tuple-ified chunk array, creates a Rectangularizer,
		and iterates over the minimal rectangularizations of the
		chunk. The _exhaustive_merge() function then finds any
		remaining rectangles that can still be merged to minimize
		codelength. Returns the minimum codelength and minimum set of
		rectangles.
		'''
		chunk_array = np.array(chunk_array, dtype=bool) # untuple the array
		rectangles = self._initial_dissection(chunk_array)
		self._best_merge_codelength = self._sum_codelength(rectangles)
		self._best_merge_rectangles = rectangles
		if len(rectangles) <= self.max_exhaustive_size:
			self._exhaustive_merge(rectangles)
			self._exhaustive_merge.cache_clear()
		else:
			self._beam_merge(rectangles)
			self._beam_merge.cache_clear()
		return self._best_merge_codelength, self._best_merge_rectangles

	def _compress_chunk(self, chunk_array, return_rectangles=True):
		'''
		Takes a chunk in the form of a 2D Boolean array describing
		where the members of the chunk are. Returns the minimum
		coding length and the set of rectangles that were found to
		minimize the coding length. A chunk is a set of items that
		are connected vertically or horizontally. For example, a
		chunk made up of five units in a 3x3 space might look like
		this:

		  [[True,  True,  True],       0 0 0
		   [False, False, True],   =   - - 0
		   [False, False, True]]       - - 0

		The method first checks for a trivial solution (the chunk
		already forms a rectangle). Otherwise, the chunk is passed
		over to the _rectangularize() method for rectangularization.
		Passing the clipped chunk to _rectangularize() increases the
		chance of a hit on the cache.
		'''
		clipped_chunk_array, rmin, cmin = self._clip_chunk(chunk_array)
		if clipped_chunk_array.all(): # the chunk already forms a rectangle
			size = clipped_chunk_array.shape
			chunk_codelength = self.codelength_lookup[size]
			chunk_rectangles = [((0, 0), (0, size[1]), (size[0], 0), size, size)]
		else:
			precomputed_solution = self._check_for_precomputed_solution(clipped_chunk_array)
			if precomputed_solution:
				chunk_codelength, chunk_rectangles = precomputed_solution
			else:
				# Chunk is first tuple-ified to allow for cacheing
				clipped_chunk_tuple = tuple([tuple(row) for row in clipped_chunk_array])
				chunk_codelength, chunk_rectangles = self._rectangularize(clipped_chunk_tuple)
		if return_rectangles:
			return chunk_codelength, self._unclip_chunk(chunk_rectangles, rmin, cmin)
		return chunk_codelength

	##################
	# PUBLIC METHODS #
	##################

	def complexity(self, language_array):
		'''
		Takes a language in the form of a 2D int array, where each
		integer represents the members of a concept. For example, a
		three-concept language in a 3x3 space might look like this:

		  [[0, 0, 0],       0 0 0
		   [1, 1, 0],   =   1 1 0
		   [0, 2, 0]]       0 2 0

		Returns the minimum codelength (complexity/compressibility)
		of that language.
		'''
		if not isinstance(language_array, np.ndarray):
			raise ValueError('language_array should be Numpy array')
		if language_array.shape != self.shape:
			raise ValueError('language_array shape %s does not match space shape %s' % (str(language_array.shape), str(self.shape)))
		codelength = 0.0
		for concept_label in np.unique(language_array):
			concept_array = language_array == concept_label
			chunkatory, n_chunks = ndimage.label(concept_array)
			for chunk_label in range(1, n_chunks+1):
				chunk_array = chunkatory == chunk_label
				codelength += self._compress_chunk(chunk_array, False)
		return codelength

	def compress_concept(self, concept_array):
		'''
		Takes a concept in the form of a 2D Boolean array describing
		where the members of the concept are. For example, a concept
		consisting of six members in a 3x3 space might look like this:

		  [[True,  True,  True],       1 1 1
		   [False, False, True],   =   0 0 1
		   [True,  False, True]]       1 0 1

		Returns the minimum coding length and the set of rectangles
		that were found to minimize the coding length.
		'''
		if not isinstance(concept_array, np.ndarray) or concept_array.dtype != bool:
			raise ValueError('concept_array should be Boolean Numpy array')
		if concept_array.shape != self.shape:
			raise ValueError('concept_array shape %s does not match space shape %s' % (str(concept_array.shape), str(self.shape)))
		concept_codelength = 0.0
		concept_rectangles = []
		chunkatory, n_chunks = ndimage.label(concept_array)
		for chunk_label in range(1, n_chunks+1):
			chunk_array = chunkatory == chunk_label
			chunk_codelength, chunk_rectangles = self._compress_chunk(chunk_array, True)
			concept_codelength += chunk_codelength
			concept_rectangles.extend(chunk_rectangles)
		return concept_codelength, concept_rectangles

	def compress_language(self, language_array):
		'''
		Takes a language in the form of a 2D int array, where each
		integer represents the members of a concept. For example, a
		three-concept language in a 3x3 space might look like this:

		  [[0, 0, 0],       0 0 0
		   [1, 1, 0],   =   1 1 0
		   [0, 2, 0]]       0 2 0

		Returns the minimum coding length and the set of rectangles
		that were found to minimize the coding length.
		'''
		if not isinstance(language_array, np.ndarray):
			raise ValueError('language_array should be Numpy array')
		if language_array.shape != self.shape:
			raise ValueError('language_array shape %s does not match space shape %s' % (str(language_array.shape), str(self.shape)))
		language_codelength = 0.0
		language_rectangles = []
		for concept_label in np.unique(language_array):
			concept_array = language_array == concept_label
			concept_codelength, concept_rectangles = self.compress_concept(concept_array)
			language_codelength += concept_codelength
			language_rectangles.append(concept_rectangles)
		return language_codelength, language_rectangles

	def encode_concept(self, concept_array):
		'''
		Takes a concept in the form of a 2D Boolean array and returns
		a binary string representing that concept.
		'''
		if not isinstance(concept_array, np.ndarray) or concept_array.dtype != bool:
			raise ValueError('concept_array should be Boolean Numpy array')
		if concept_array.shape != self.shape:
			raise ValueError('concept_array shape %s does not match space shape %s' % (str(concept_array.shape), str(self.shape)))
		_, concept_rectangles = self.compress_concept(concept_array)
		binary_string = ''
		for rect in concept_rectangles:
			binary_string += self.symbol_to_string[rect[4][0], rect[4][1], rect[0][0], rect[0][1]]
		return binary_string

	def decode_concept(self, binary_string):
		'''
		Takes a binary string and resolves it into a 2D Boolean array
		representing the concept described by that binary string.
		'''
		if not isinstance(binary_string, str) or not all(digit in ['0','1'] for digit in binary_string):
			raise ValueError('binary_string should be a string of 0s and 1s')
		concept_array = np.zeros(self.shape, dtype=bool)
		last_split_point = 0
		for split_point in range(1, len(binary_string)+1):
			substring = binary_string[last_split_point:split_point]
			if substring in self.string_to_symbol:
				rect = self.string_to_symbol[substring]
				y_start, y_end = rect[2], rect[2]+rect[0]
				x_start, x_end = rect[3], rect[3]+rect[1]
				if concept_array[y_start:y_end, x_start:x_end].any() == True:
					raise ValueError('Invalid binary string: The substring %s results in rectangular overlap' % substring)
				concept_array[y_start:y_end, x_start:x_end] = True
				last_split_point = split_point
		if last_split_point != len(binary_string):
			raise ValueError('Cannot resolve binary string: %s![%s]' % (binary_string[:last_split_point], binary_string[last_split_point:]))
		return concept_array

	def encode_language(self, language_array):
		'''
		Takes a language in the form of a 2D int array and returns a
		set of binary strings, each representing one of the concepts
		in the language.
		'''
		if not isinstance(language_array, np.ndarray):
			raise ValueError('language_array should be Numpy array')
		if language_array.shape != self.shape:
			raise ValueError('language_array shape %s does not match space shape %s' % (str(language_array.shape), str(self.shape)))
		binary_strings = []
		for concept_label in np.unique(language_array):
			concept_array = language_array == concept_label
			binary_string = self.encode_concept(concept_array)
			binary_strings.append(binary_string)
		return binary_strings

	def decode_language(self, binary_strings):
		'''
		Takes a list of binary strings and returns a 2D int array
		representing the language (set of concepts) described by
		those binary strings.
		'''
		if not isinstance(binary_strings, list):
			raise ValueError('binary_strings should be a list of binary strings')
		language_array = np.full(self.shape, -1, dtype=int)
		for concept_i, binary_string in enumerate(binary_strings):
			concept_array = self.decode_concept(binary_string)
			language_array[np.where(concept_array == True)] = concept_i
		if language_array.min() == -1:
			raise ValueError('These binary strings do not resolve to a complete language')
		return language_array

	def tabulate(self, show_symbols=False):
		'''
		Prints a tabulation showing the probabilities and codelengths
		of each rectangle class, or the codelengths and (non-optimal)
		Shannon-Fano codewords for the rectangle symbols.
		'''
		from tabulate import tabulate
		table = []
		n_shapes = len(self.rectangle_shapes)
		for rect_class in self.rectangle_shapes:
			n_locations = len(self.rectangle_symbols[rect_class])
			codelength = self.codelength_lookup[rect_class]
			probability = 2 ** -codelength
			if show_symbols:
				for rect in self.rectangle_symbols[rect_class]:
					table.append(['%ix%i'%(rect[0], rect[1]), '%i,%i'%(rect[2], rect[3]), '-log 1/%i = %s'%(n_shapes*n_locations, str(round(codelength, 5))), self.symbol_to_string[rect]])
			else:
				table.append(['%ix%i'%rect_class, '%i'%n_locations, '1/%i x 1/%i = %s'%(n_shapes, n_locations, str(round(probability, 5))), '-log 1/%i = %s'%(n_shapes*n_locations, str(round(codelength, 5)))])
		if show_symbols:
			print(tabulate(table, headers=['Class', 'Location', 'Codelength (bits)', 'Shannon-Fano codeword'], numalign="left"))
		else:
			print(tabulate(table, headers=['Class', 'N locations', 'Probability', 'Codelength (bits)'], numalign="left"))
