'''
INCOMPLETE: These are the beginnings of a module that allows you to
iterate over all dissections of a recilinear polygon into the minimum
number of rectangles. In its current state, it performs fairly well
but there are still edge cases that it can't deal with. This was
intended to be a module imported by rectlang.py, which would use it to
iterate overal minimal dissections and then choose the one that
minimizes codelength. However, since I didn't complete this,
rectlang.py still relies on slower methods.
'''

from collections import defaultdict
from itertools import combinations, permutations, product
from functools import lru_cache
from scipy import ndimage
import numpy as np

hole_struct = np.ones((3, 3), dtype=int)

class Vertex:

	def __init__(self, point, concave=False, index=None, nxt=None):
		self.yx = point
		self.y, self.x = point
		self.concave = concave
		self.index = index
		if nxt is None and self.index is not None:
			nxt = self.index + 1
		self.nxt = nxt
		self.vertical_chord_dir = None
		self.horizontal_chord_dir = None
		self.visited = False
		self.chord_index = None

	def __repr__(self):
		convexity = 'v' if self.concave else 'x'
		return 'Vertex' + str(self.yx) + convexity

	def __eq__(self, item):
		if self.yx == item.yx:
			return True
		return False

	def move_vertically(self):
		y = self.y
		if self.vertical_chord_dir == 'down':
			while True:
				y += 1
				yield Vertex((y, self.x))
		if self.vertical_chord_dir == 'up':
			while True:
				y -= 1
				yield Vertex((y, self.x))

	def move_horizontally(self):
		x = self.x
		if self.horizontal_chord_dir == 'right':
			while True:
				x += 1
				yield Vertex((self.y, x))
		if self.horizontal_chord_dir == 'left':
			while True:
				x -= 1
				yield Vertex((self.y, x))

class Edge:

	def __init__(self, vertices):
		self.vertices = vertices
		self.start = self.vertices[0]
		self.end = self.vertices[-1]
		if self.start.y == self.end.y:
			self.vertical = False
			self.horizontal = True
			if self.start.x < self.end.x:
				self.direction = 'right'
			else:
				self.direction = 'left'
		else:
			self.vertical = True
			self.horizontal = False
			if self.start.y < self.end.y:
				self.direction = 'down'
			else:
				self.direction = 'up'

	def __iter__(self):
		for vertex in self.vertices:
			yield vertex

	def __contains__(self, element):
		if element.start in self.vertices and element.end in self.vertices:
			return True
		return False

	def __repr__(self):
		m = ' > ' if self.horizontal else ' ^ '
		return 'Edge[' + str(self.start) + m + str(self.end) + ']'

class Chord:

	def __init__(self, start, end):
		self.start = start
		self.end = end
		if start.x == end.x:
			self.vertical_orientation = True
		else:
			self.vertical_orientation = False
		self.min_x, self.max_x = sorted([self.start.x, self.end.x])
		self.min_y, self.max_y = sorted([self.start.y, self.end.y])
		self.sorted_start, self.sorted_end = sorted([self.start.yx, self.end.yx])
		self._hash = hash((self.sorted_start, self.sorted_end))
		self.matched = False

	def __iter__(self):
		start, end = sorted([self.start.yx, self.end.yx])
		if self.vertical_orientation:
			for i in range(start[0]+1, end[0]):
				yield Vertex((i, start[1]))
		else:
			for i in range(start[1]+1, end[1]):
				yield Vertex((start[0], i))

	def __eq__(self, chord):
		if chord.sorted_start != self.sorted_start:
			return False
		if chord.sorted_end != self.sorted_end:
			return False
		return True

	def __hash__(self):
		return self._hash

	def __repr__(self):
		return 'Chord[' + str(self.start) + ' > ' + str(self.end) + ']'

	def intersection_point(self, chord):
		'''
		If chords have a non-terminal intersection point, return it, else return False
		'''
		if self.vertical_orientation != chord.vertical_orientation:
			if self.vertical_orientation:
				if chord.start.y > self.min_y and chord.start.y < self.max_y and self.start.x > chord.min_x and self.start.x < chord.max_x:
					return Vertex((chord.start.y, self.start.x))
			else:
				if chord.start.x > self.min_x and chord.start.x < self.max_x and self.start.y > chord.min_y and self.start.y < chord.max_y:
					return Vertex((self.start.y, chord.start.x))
		return False

	def other_end(self, vertex):
		if vertex == self.start:
			return self.end
		else:
			return self.start

class Outline:

	def __init__(self, edges, points):
		self.edges = edges
		self.points = points
		self._hash = hash(tuple([point.yx for point in self.points]))

	def __iter__(self):
		for edge in self.edges:
			yield edge

	def __hash__(self):
		return self._hash

	def get_next_vert(self, vert):
		if vert.nxt is None:
			vert = self.points[self.points.index(vert)]
		return self.points[vert.nxt]

class Rectangularization:

	def __init__(self, outline, chord_set, chunk_array):
		self.outline = outline
		self.chunk_array = chunk_array
		self.chord_set = self._clip_chords(chord_set)
		self.chord_map = self._make_chord_map()
		self.rects = self._identify_rects()

	def __iter__(self):
		'''
		Iterating over a rectangularization yields each of the rectangles it is composed of.
		'''
		for rect in self.rects:
			yield rect

	def __repr__(self):
		'''
		Returns an ASCII representation of the rectangularization.
		'''
		s = [[' ' for _ in range(10)] for __ in range(10)]
		for edge in self.outline:
			line_marker = '|' if edge.vertical else '-'
			for point in edge:
				s[point.y][point.x] = line_marker
			s[edge.start.y][edge.start.x] = 'v' if edge.start.concave else 'x'
			s[edge.end.y][edge.end.x] = 'v' if edge.end.concave else 'x'
		for chord in self.chord_set:
			for point in chord:
				s[point.y][point.x] = '.'
		return '  ' + ' '.join(map(str, range(len(s[0])))) + '\n' + '\n'.join([str(y) + ' ' + ' '.join(row) for y, row in enumerate(s)])

	def draw(self, svg_file_name):
		'''
		Makes an SVG drawing of the rectangularization.
		'''
		svg = "<svg width='5in' height='5in' viewBox='0 0 10 10' xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#' xmlns:svg='http://www.w3.org/2000/svg' xmlns='http://www.w3.org/2000/svg' version='1.1'>\n\n"
		for y in range(1,9):
			svg += '<text text-anchor="middle" x="0.5" y="%i" fill="black" style="font-size: 0.5px; font-family:Helvetica;">%s</text>\n' %(y, str(y))
		for x in range(1,9):
			svg += '<text text-anchor="middle" x="%i" y="0.5" fill="black" style="font-size: 0.5px; font-family:Helvetica;">%s</text>\n' %(x, str(x))
		points = ' '.join([str(vert.x) + ',' + str(vert.y) for vert in self.outline.points])
		svg += '<polygon points="%s" style="stroke: black; stroke-width:0.1; fill:white;" />\n' % points
		for (y,x), val in np.ndenumerate(self.chunk_array):
			if val:
				svg += '<circle cx="%f" cy="%f" r="0.2" style="stroke-width:0; fill:black;" />\n' %(x+1.5, y+1.5)
		for chord in self.chord_set:
			svg += '<line x1="%i" y1="%i" x2="%i" y2="%i" stroke-width="0.1" stroke="red"/>' % (chord.start.x, chord.start.y, chord.end.x, chord.end.y)
		for rect in self.rects:
			height = rect[1][0] - rect[0][0]
			width = rect[1][1] - rect[0][1]
			svg += '<rect x="%f" y="%f" width="%f" height="%f" style="fill:blue;stroke-width:0;fill-opacity:0.1;stroke-opacity:0.9" />' % (rect[0][1]+0.2, rect[0][0]+0.2, width-0.4, height-0.4)
		svg += '</svg>'
		with open(svg_file_name, mode='w') as file:
			file.write(svg)

	def _clip_chords(self, chord_set):
		'''
		Takes a set of chords in a particular order and checks if
		later chords intersect with earlier ones. If so, both chords
		are clipped at the intersection point.
		'''
		if len(chord_set) < 2:
			return chord_set
		made_a_clipping = True
		while made_a_clipping:
			made_a_clipping = False
			clipped_chords = []
			for chord in chord_set:
				intersection_points = []
				for match_i, m_chord in enumerate(clipped_chords):
					intersection = chord.intersection_point(m_chord)
					if intersection:
						intersection_points.append((intersection, match_i))
				if len(intersection_points) > 0:
					clipped_end_point, match_i = closest_intersection(chord.start, intersection_points)
					s = clipped_chords[match_i].start
					e = clipped_chords[match_i].end
					split1 = Chord(s, clipped_end_point)
					split2 = Chord(e, clipped_end_point)
					del clipped_chords[match_i]
					clipped_chords.append(split1)
					clipped_chords.append(split2)
					clipped_chords.append(Chord(chord.start, clipped_end_point))
					if chord.end.concave:
						clipped_chords.append(Chord(chord.end, clipped_end_point))
					made_a_clipping = True
				else:
					clipped_chords.append(chord)
			chord_set = clipped_chords
		return clipped_chords

	def _make_chord_map(self):
		'''
		Creates a dictionary mapping chord start and end points to
		the chords that eminate from them.
		'''
		chord_map = defaultdict(list)
		for chord in self.chord_set:
			chord_map[chord.start.yx].append(chord)
			chord_map[chord.end.yx].append(chord)
		return chord_map

	def _identify_rects(self):
		'''
		Identifies all the rectangles that make up the chunk, given
		the current chord set.
		'''
		rects = self._traverse_outline()
		test_array = np.zeros(self.chunk_array.shape, dtype=bool)
		for tl, br in rects:
			test_array[tl[0]-1:br[0]-1, tl[1]-1:br[1]-1] = True
		if not np.array_equal(test_array, self.chunk_array):
			remainder_array = self.chunk_array ^ test_array
			remaining_rects = self._identify_remaining_rects(remainder_array)
			rects.extend(remaining_rects)
		return rects

	def _traverse_outline(self):
		'''
		Traverses the outline, following chords as necessary, in
		order to identify the loops (rectangles) that the chunk is
		comprised of.
		'''
		self.outline.points[0].visited = True
		loops = []
		for vert in self.outline.points[1:]:
			if vert.visited:
				continue
			last_vert = self.outline.points[vert.index - 1]
			loop = [last_vert]
			stop_bounce_back_on_this_chord = None
			while len(loop) < 4 or vert not in loop:
				loop.append(vert)
				vert.visited = True
				if vert.yx in self.chord_map:
					chords = self.chord_map[vert.yx]
					if stop_bounce_back_on_this_chord:
						chords = [chord for chord in chords if chord != stop_bounce_back_on_this_chord]
						stop_bounce_back_on_this_chord = None
					if len(chords) == 0:
						vert = self.outline.get_next_vert(vert)
					else:
						chord = self.select_ccw_chord(loop[-2], vert, chords)
						if chord:
							stop_bounce_back_on_this_chord = chord
							vert = chord.other_end(vert)
						else:
							vert = self.outline.get_next_vert(vert)
				else:
					vert = self.outline.get_next_vert(vert)
			loops.append(loop[1:])
		rects = list({self._measure_rect(loop) for loop in loops})
		for vert in self.outline.points:
			vert.visited = False
		return rects

	def _identify_remaining_rects(self, remainder_array):
		'''
		If the test indicated that not all elements in the chunk
		array were identified, try to identify the remainder. First
		label contiguous chunks in the remainder array. Then, for
		each contiguous chunk, first check for a trivial solution,
		and if that's not possible, create a subrectangularization
		that inherits this rectangularization's chords and get it's
		rects.
		'''
		remaining_rects = []
		giblets, n_giblets = ndimage.label(remainder_array)
		for giblet_label in range(1, n_giblets+1):
			giblet_array = giblets == giblet_label
			rmin, rmax = np.where(np.any(giblet_array, axis=1))[0][[0, -1]]
			cmin, cmax = np.where(np.any(giblet_array, axis=0))[0][[0, -1]]
			clipped_giblet_array = giblet_array[rmin:rmax+1, cmin:cmax+1]
			if clipped_giblet_array.all():
				remaining_rects.append(((rmin+1, cmin+1), (rmax+2, cmax+2)))
			else:
				outline = array_to_outline(clipped_giblet_array)
				chords = []
				for chord in self.chord_set:
					try:
						s = outline.points.index(chord.start)
						e = outline.points.index(chord.end)
					except ValueError:
						continue
					ignore_this_chord = False
					for edge in outline:
						if chord in edge:
							ignore_this_chord = True
							break
					if not ignore_this_chord:
						new_chord = Chord(outline.points[s], outline.points[e])
						chords.append(new_chord)
				rectangularization = create_rectangularization(outline, chords, clipped_giblet_array)
				for rect in rectangularization:
					rect = ((rect[0][0]+rmin, rect[0][1]+cmin), (rect[1][0]+rmin, rect[1][1]+cmin))
					remaining_rects.append(rect)
		return remaining_rects

	def _measure_rect(self, loop):
		'''
		Takes a list of points that form a loop and finds the min and
		max x- and y-coordinates.
		'''
		lo = [ 999999,  999999]
		hi = [-999999, -999999]
		for vert in loop:
			lo[0] = min(vert.y, lo[0])
			hi[0] = max(vert.y, hi[0])
			lo[1] = min(vert.x, lo[1])
			hi[1] = max(vert.x, hi[1])
		return (tuple(lo), tuple(hi))

	def select_ccw_chord(self, prev_point, curr_point, chords):
		'''
		Given a selection of chords, choose one that turns counterclockwise or continues straight on
		'''
		if prev_point.x == curr_point.x: #vertical
			if prev_point.y < curr_point.y: #down
				for choice in chords:
					if choice.other_end(curr_point).x > curr_point.x:
						return choice
				for choice in chords:
					if choice.other_end(curr_point).x == curr_point.x and choice.other_end(curr_point).y > curr_point.y:
						return choice
			else: #up
				for choice in chords:
					if choice.other_end(curr_point).x < curr_point.x:
						return choice
				for choice in chords:
					if choice.other_end(curr_point).x == curr_point.x and choice.other_end(curr_point).y < curr_point.y:
						return choice
		else: #horizontal
			if prev_point.x < curr_point.x: #right
				for choice in chords:
					if choice.other_end(curr_point).y < curr_point.y:
						return choice
				for choice in chords:
					if choice.other_end(curr_point).y == curr_point.y and choice.other_end(curr_point).x > curr_point.x:
						return choice
			else: #left
				for choice in chords:
					if choice.other_end(curr_point).y > curr_point.y:
						return choice
				for choice in chords:
					if choice.other_end(curr_point).y == curr_point.y and choice.other_end(curr_point).x < curr_point.x:
						return choice
		return False


def closest_intersection(point, intersection_points):
	'''
	Given a point and a list of possible intersection points, choose
	the closest one.
	'''
	dists = []
	for point2, match_i in intersection_points:
		dists.append(ED(point, point2))
	return intersection_points[np.argmin(dists)]

def ED(point1, point2):
	'''
	Returns square Euclidean distance between two points.
	'''
	return (point1.y - point2.y)**2 + (point1.x - point2.x)**2

@lru_cache(maxsize=2**10)
def binary_sequences(length):
	'''
	Returns list of all binary sequences of given length. Each
	concave vertex has two possible chords eminating from it, so this
	is used to enumerate all combinations of chords. The function is
	cached for fast retrieval.
	'''
	if length == 0:
		return [[]]
	start = [0] * length
	sequences = [start]
	for sequence in next_binary_sequence(0, start):
		sequences.append(sequence)
	return sequences

def next_binary_sequence(i, sequence):
   b = sequence[:i] + [1] + sequence[i + 1:]
   yield b
   if i + 1 < len(sequence):
	   for next_sequence in next_binary_sequence(i + 1, sequence):
		   yield next_sequence
	   for next_sequence in next_binary_sequence(i + 1, b):
		   yield next_sequence


class Rectangularizer:

	def __init__(self, chunk_array):
		self.chunk_array = chunk_array
		self.outline = array_to_outline(chunk_array)
		self.first_hole_index = len(self.outline.points)
		self.holes = self._find_holes()
		self._append_hole_vertices_to_outline()
		self.fixed_chords, self.unfixed_chords, self.min_chords = self._find_chords()
		print('Minimum chords:', self.min_chords)

	def __repr__(self):
		'''
		Returns an ASCII representation of the chunk.
		'''
		s = [[' ' for _ in range(10)] for __ in range(10)]
		for edge in self.outline:
			line_marker = '|' if edge.vertical else '-'
			for point in edge:
				s[point.y][point.x] = line_marker
			s[edge.start.y][edge.start.x] = 'v' if edge.start.concave else 'x'
			s[edge.end.y][edge.end.x] = 'v' if edge.end.concave else 'x'
		for hole in self.holes:
			for edge in hole:
				line_marker = '|' if edge.vertical else '-'
				for point in edge:
					s[point.y][point.x] = line_marker
				s[edge.start.y][edge.start.x] = 'v' if edge.start.concave else 'x'
				s[edge.end.y][edge.end.x] = 'v' if edge.end.concave else 'x'
		return '  ' + ' '.join(map(str, range(len(s[0])))) + '\n' + '\n'.join([str(y) + ' ' + ' '.join(row) for y, row in enumerate(s)])

	# def __iter__(self):
	# 	'''
	# 	Iterates over rectangularizations of the chunk that are
	# 	comprised of the minimum number of rectangles.
	# 	'''
	# 	sequences = []
	# 	for sequence in binary_sequences(len(self.chords)):
	# 		print(sequence)
	# 		chords = set()
	# 		for chord_set_i, choice in enumerate(sequence):
	# 			chords.add(self.chords[chord_set_i][choice])
	# 		sequences.append(list(chords))
	# 	min_chords = min([len(chord_set) for chord_set in sequences]) + 1
	# 	for sequence in sequences:
	# 		if len(sequence) <= min_chords:
	# 			for alt_sequence in self._intersection_alternates(sequence):
	# 				yield create_rectangularization(self.outline, alt_sequence, self.chunk_array)


	def __iter__(self):
		for sequence in product(*self.unfixed_chords):
			sequence = self.fixed_chords + list(set(sequence))
			if len(sequence) == self.min_chords:
				rec = create_rectangularization(self.outline, sequence, self.chunk_array)
				yield rec

	def _find_holes(self):
		'''
		Identify holes in the chunk array, label each hole, and then
		create an outline for each hole.
		'''
		hole_outlines = []
		fill_complement = ndimage.morphology.binary_fill_holes(self.chunk_array, hole_struct) ^ self.chunk_array
		holes, n_holes = ndimage.label(fill_complement)
		for hole_label in range(1, n_holes+1):
			hole_array = holes == hole_label
			hole_outline = array_to_outline(hole_array, hole=True)
			hole_outlines.append(hole_outline)
		return hole_outlines

	def _append_hole_vertices_to_outline(self):
		'''
		Iterate over the holes and add each hole's vertices into the
		outline. The hole's vertices are given indexes that continue
		from where the outline finished, but the nxt indices look
		backwards such that holes will be traversed in clockwise
		order.
		'''
		index = self.first_hole_index
		for hole in self.holes:
			hole_points = []
			for point in hole.points[:-1]:
				point.index = index
				point.nxt = index - 1
				hole_points.append(point)
				index += 1
			hole_points[0].nxt = index - 1
			self.outline.points.extend(hole_points)

	def _intersection_alternates(self, sequence):
		if len(sequence) == 1:
			yield sequence
		else:
			matched = set()
			unmatched_intersecting = set()
			for chord_i, chord_j in combinations(sequence, 2):
				if chord_i.intersection_point(chord_j):
					if chord_i.matched:
						matched.add(chord_i)
					else:
						unmatched_intersecting.add(chord_i)
					if chord_j.matched:
						matched.add(chord_j)
					else:
						unmatched_intersecting.add(chord_j)
			unmatched_non_intersecting = set()
			for chord in sequence:
				if chord not in matched and chord not in unmatched_intersecting:
					unmatched_non_intersecting.add(chord)
			fixed_set = list(matched) + list(unmatched_non_intersecting)
			for perm in permutations(unmatched_intersecting):
				yield fixed_set + list(perm)

	def _get_concave_vertices(self):
		'''
		Iterate over the outline and create a list of all the concave
		vertices. The concave verticies from holes will be added
		later.
		'''
		concave_vertices = []
		for edge in self.outline:
			if edge.start.concave:
				concave_vertices.append(edge.start)
		return concave_vertices

	def _find_chords(self):
		'''
		Iterates over each concave vertex and finds the vertical and
		horizontal chords eminating from it.
		'''
		chord_pairs = []
		for vert in self.outline.points:
			if not vert.concave:
				continue
			chord_pair = []
			match_count = 0
			for mover in (vert.move_vertically, vert.move_horizontally):
				for candidate in mover():
					try:
						end_vert = self.outline.points[self.outline.points.index(candidate)]
					except ValueError:
						continue
					break
				chord = Chord(vert, end_vert)
				if end_vert.concave:
					chord.matched = True
					match_count += 1
				chord_pair.append(chord)
			vert.chord_index = match_count
			chord_pairs.append((chord_pair, match_count))


		fixed_chords = set()
		unfixed_chords = []
		for chord_pair, match_count in chord_pairs:
			if match_count == 1:
				if chord_pair[0].matched:
					if chord_pair[0].end.chord_index == 1:
						fixed_chords.add(chord_pair[0])
						continue
				else:
					if chord_pair[1].end.chord_index == 1:
						fixed_chords.add(chord_pair[1])
						continue
			unfixed_chords.append(chord_pair)

		count = 0
		seen = set()
		for chord_pair in unfixed_chords:
			if chord_pair[0] in seen:
				count += 1
			else:
				seen.add(chord_pair[0])
			if chord_pair[1] in seen:
				count += 1
			else:
				seen.add(chord_pair[1])

		return list(fixed_chords), unfixed_chords, len(fixed_chords)+len(unfixed_chords)-int(np.ceil(count/2))

def array_to_outline(chunk_array, hole=False):
	'''
	Takes a Boolean array representing a contiguous chunk and draws
	an outline around the area that is True. If hole is set to True,
	convex vertices are marked as concave and the chord directions
	are flipped.
	'''
	vertex_set, vertex_mapper = identify_vertices(chunk_array)
	ordered_vertices = order_vertices(vertex_set, vertex_mapper, hole)
	outline = create_outline(ordered_vertices, hole)
	return outline

def identify_vertices(chunk_array):
	'''
	Scans the chunk_array looking for adjacent cells that are unequal
	to build a set of vertices around the edge of the chunk.
	'''
	height, width = chunk_array.shape
	chunk_array_no_holes = ndimage.morphology.binary_fill_holes(chunk_array, hole_struct)
	padded_chunk_array = np.zeros((height+2, width+2))
	padded_chunk_array[1:height+1, 1:width+1] = chunk_array_no_holes
	vertex_set = set()
	vertex_mapper = defaultdict(list)
	for y in range(padded_chunk_array.shape[0]-1):
		y_ = y + 1
		for x in range(padded_chunk_array.shape[1]-1):
			x_ = x + 1
			if padded_chunk_array[y,x] != padded_chunk_array[y_,x]:
				vertex_set.add((y_,x))
				vertex_set.add((y_,x_))
				vertex_mapper[(y_,x)].append((y_,x_))
				vertex_mapper[(y_,x_)].append((y_,x))
			if padded_chunk_array[y,x] != padded_chunk_array[y,x_]:
				vertex_set.add((y,x_))
				vertex_set.add((y_,x_))
				vertex_mapper[(y,x_)].append((y_,x_))
				vertex_mapper[(y_,x_)].append((y,x_))
	vertex_set = sorted(list(vertex_set))
	return vertex_set, vertex_mapper

def order_vertices(vertex_set, vertex_mapper, hole=False):
	'''
	Starting from the two topmost leftmost vertices, move in a
	counterclockwise direction to order the vertices CCW.
	'''
	vertices = [Vertex(vertex_set[1], index=0), Vertex(vertex_set[0], index=1)]
	index = 1
	while vertices[0] != vertices[-1]:
		index += 1
		cur = vertices[-1]
		prv = vertices[-2]
		if cur.x < prv.x: #left
			nxt = cur.y+1, cur.x #down
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				if hole:
					vertices[-1].concave = True
					vertices[-1].vertical_chord_dir = 'up'
					vertices[-1].horizontal_chord_dir = 'left'
				vertices.append(nxt)
				continue
			nxt = cur.y, cur.x-1 #left
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				vertices.append(nxt)
				continue
			nxt = cur.y-1, cur.x #up
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				if not hole:
					vertices[-1].concave = True
					vertices[-1].vertical_chord_dir = 'down'
					vertices[-1].horizontal_chord_dir = 'left'
				vertices.append(nxt)
				continue
		if cur.y > prv.y: #down
			nxt = cur.y, cur.x+1 #right
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				if hole:
					vertices[-1].concave = True
					vertices[-1].vertical_chord_dir = 'down'
					vertices[-1].horizontal_chord_dir = 'left'
				vertices.append(nxt)
				continue
			nxt = cur.y+1, cur.x #down
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				vertices.append(nxt)
				continue
			nxt = cur.y, cur.x-1 #left
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				if not hole:
					vertices[-1].concave = True
					vertices[-1].vertical_chord_dir = 'down'
					vertices[-1].horizontal_chord_dir = 'right'
				vertices.append(nxt)
				continue
		if cur.x > prv.x: #right
			nxt = cur.y-1, cur.x #up
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				if hole:
					vertices[-1].concave = True
					vertices[-1].vertical_chord_dir = 'down'
					vertices[-1].horizontal_chord_dir = 'right'
				vertices.append(nxt)
				continue
			nxt = cur.y, cur.x+1 #right
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				vertices.append(nxt)
				continue
			nxt = cur.y+1, cur.x #down
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				if not hole:
					vertices[-1].concave = True
					vertices[-1].vertical_chord_dir = 'up'
					vertices[-1].horizontal_chord_dir = 'right'
				vertices.append(nxt)
				continue
		if cur.y < prv.y: #up
			nxt = cur.y, cur.x-1 #left
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				if hole:
					vertices[-1].concave = True
					vertices[-1].vertical_chord_dir = 'up'
					vertices[-1].horizontal_chord_dir = 'right'
				vertices.append(nxt)
				continue
			nxt = cur.y-1, cur.x #up
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				vertices.append(nxt)
				continue
			nxt = cur.y, cur.x+1 #right
			if nxt in vertex_mapper[cur.yx]:
				nxt = Vertex(nxt, index=index)
				if not hole:
					vertices[-1].concave = True
					vertices[-1].vertical_chord_dir = 'up'
					vertices[-1].horizontal_chord_dir = 'left'
				vertices.append(nxt)
				continue
	vertices[-1].nxt = 1
	return vertices

def create_outline(ordered_vertices, hole=False):
	'''
	Traverse the ordered vertices to build up a set of edges. When
	the direction changes mark the vertex as concave if it's concave.
	'''
	edges = []
	edge = [ordered_vertices[1]]
	vertical_orientation = True
	for vert in ordered_vertices[2:]:
		if vertical_orientation:
			if edge[-1].x != vert.x:
				edges.append(Edge(edge))
				edge = [edge[-1], vert]
				vertical_orientation = False
			else:
				edge.append(vert)
		else:
			if edge[-1].y != vert.y:
				edges.append(Edge(edge))
				edge = [edge[-1], vert]
				vertical_orientation = True
			else:
				edge.append(vert)
	edge.append(ordered_vertices[1])
	edges.append(Edge(edge))
	if edges[-1].start.y != edges[-1].end.y:
		if hole:
			ordered_vertices[0].concave = True
			ordered_vertices[0].vertical_chord_dir = 'up'
			ordered_vertices[0].horizontal_chord_dir = 'right'
			ordered_vertices[-1].concave = True
			ordered_vertices[-1].vertical_chord_dir = 'up'
			ordered_vertices[-1].horizontal_chord_dir = 'right'
		edges[-1].vertices[-1] = ordered_vertices[0]
		edges.append(Edge([ordered_vertices[0], ordered_vertices[1]]))
	return Outline(edges, ordered_vertices)

def create_rectangularization(outline, chord_set, chunk_array):
	'''
	Takes an outline and a set of chords and returns a
	rectangularization object, pulling from an LRU cache if
	possible.
	'''
	outline_hash = hash(outline)
	chord_set_hash = hash(tuple(sorted([hash(chord) for chord in chord_set])))
	rectangularization_cache.initialize(outline, chord_set, chunk_array)
	return rectangularization_cache.create(outline_hash, chord_set_hash)

class RectangularizationCache:

	'''
	Stores a cache of rectangularizations (particular outlines
	combined with particular chords). If the rectangularization
	hasn't been observed before, a new Rectangularization object is
	created, but if it has been observed before, the object is pulled
	directly from the cache. To use the cache, first call the
	initialize() method to pass in the outline, chords, and
	chunk_array, and then call the create() method with hashes of the
	outline and chord_set. The create_rectangularization() function
	does this automatically.
	'''

	def __init__(self):
		self.deinitialize()

	def initialize(self, outline, chord_set, chunk_array):
		'''
		Prepare for a call to the create() method.
		'''
		self._outline = outline
		self._chord_set = chord_set
		self._chunk_array = chunk_array
		self._ready = True

	def deinitialize(self):
		self._outline = None
		self._chord_set = None
		self._chunk_array = None
		self._ready = False

	@lru_cache(maxsize=2**20)
	def create(self, outline_hash, chord_set_hash):
		'''
		Create a new rectangularization, drawing from the cache if
		possible.
		'''
		if not self._ready:
			raise ValueError('The cache must first be initialized for use by a call to the initialize() method.')
		rectangularization = Rectangularization(self._outline, self._chord_set, self._chunk_array)
		self.deinitialize()
		return rectangularization

	def info(self):
		'''
		Returns stats about the current state of the cache.
		'''
		return self.create.cache_info()

rectangularization_cache = RectangularizationCache()

if __name__ == '__main__':

	# Pass in a Bool array like this, and watch it get broken into rects!

	square = np.array([
		[1,1,1,0,0,0,0,0],
		[1,1,1,0,0,0,0,0],
		[1,1,1,1,1,1,0,0],
		[1,1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1,1],
		[1,1,1,0,0,0,1,1],
		[1,1,1,0,0,0,0,0],
		[1,1,1,0,0,0,0,0]],
	dtype=bool)

	actual_size = square.sum()

	chunk = Rectangularizer(square)
	print(chunk)
	c = 0
	for i, rectangularization in enumerate(chunk):
		c+=1
		rectangularization.draw('/Users/jon/Desktop/im.svg')
		# break
		test = np.zeros(square.shape, dtype=int)
		test_size = 0
		for rect in rectangularization:
			tl, br = rect
			test[tl[0]-1:br[0]-1, tl[1]-1:br[1]-1] = 1
			height = br[0] - tl[0]
			width = br[1] - tl[1]
			test_size += height*width
		if not np.array_equal(test, square):
			print('MATCH ERROR')
			print(rectangularization)
			print(test == square)
			print(rectangularization.chord_set)
			print(rectangularization.rects)
			print(rectangularization.log)
		elif test_size != actual_size:
			print('SIZE ERROR', test_size, actual_size)
			print(rectangularization)
			print(test == square)
			print(rectangularization.chord_set)
			print(rectangularization.rects)
			print(rectangularization.log)
		else:
			print('THUMBS UP!!')
			print(rectangularization)
			print(rectangularization.rects)
		print('======================================================')
	print('COUNT', c)
	print(rectangularization_cache.info())
