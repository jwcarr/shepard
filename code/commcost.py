'''
Various classes and functions for handling Regier and colleagues'
communicative cost model.
'''

import numpy as np
from scipy.spatial.distance import pdist, squareform

class Partition:

	'''
	A partition object represents a partition of an n-dimensional
	space. To create a partition, pass a list like [[0,0,1,1],
	[0,0,1,1]], where the structure of the lists represents the space
	(here 2x4), and the  numbers represent the categories (here
	category 0 and 1). Passing a tuple like (2,4) creates a trivial
	partition of given dimensionality. Various iteration methods are
	available for traversing the partition.
	'''

	@property
	def shape(self):
		return self._partition.shape

	@property
	def size(self):
		return self._partition.size

	def __init__(self, partition):
		if isinstance(partition, tuple):
			self._partition = np.zeros(partition, dtype=int)
		else:
			self._partition = np.array(partition, dtype=int)
		self._boolean_matrix = None

	def __repr__(self):
		'''
		Provides textual description of the partition object.
		'''
		if len(self.shape) == 1:
			return 'Partition[length=%i, n_categories=%i]' % (self.shape[0], self.__len__())
		return 'Partition[shape=%s, n_categories=%i]' % ('x'.join(map(str, self.shape)), self.__len__())

	def __str__(self):
		'''
		Provides printable representation of the partition object.
		'''
		return self._partition.__str__()

	def __len__(self):
		'''
		The length of a partition is the number of categories it
		contains.
		'''
		return np.unique(self._partition).size

	def __getitem__(self, key):
		'''
		Pass a tuple to get the category memebership of a point. Pass
		an integer to get a list of points that belong to a category.
		'''
		if isinstance(key, tuple):
			return self._partition[key]
		return list(map(tuple, np.transpose(np.where(self._partition==key))))

	def __setitem__(self, key, value):
		'''
		Change the category membership of a particular point.
		'''
		if not isinstance(key, tuple):
			raise ValueError('Index must be tuple. For 1D spaces, include a trailing comma in the index.')
		self._boolean_matrix = None
		self._partition[key] = value

	def __iter__(self):
		'''
		Default iterator. Each iteration returns a point in the space
		along with its associated category.
		'''
		for point, category in np.ndenumerate(self._partition):
			yield point, category

	def iter_categories(self):
		'''
		Iterate over categories in the partition. Each iteration
		returns an integer.
		'''
		for category in np.unique(self._partition):
			yield category

	def iter_points(self):
		'''
		Iterate over points in the space. Each iteration returns a
		tuple.
		'''
		for point in np.ndindex(self.shape):
			yield point

	def boolean_matrix(self):
		'''
		Returns a 2D Boolean matrix, where rows correspond to meanings
		and columns correspond to categories. True indicates that the
		ith meaning belongs to the jth category. This Boolean matrix
		representation is used by the communicative_cost method in the
		Space object for fast computation using a similarity matrix.
		'''
		if self._boolean_matrix:
			return self._boolean_matrix
		self._boolean_matrix = convert_to_bool_matrix(self._partition)
		return self._boolean_matrix

	def spawn_speaker(self):
		'''
		Creates a Speaker with perfect speaker certainty.
		'''
		return Speaker(self.shape)

	def spawn_listener(self, gamma, mu=2):
		'''
		Creates a Listener who represents the partition according to
		the specified gamma and mu parameters. gamma may be set to
		'uniform' to create a uniform listener.
		'''
		return Listener(self.shape, self.listener_distributions(gamma, mu))

	def listener_distributions(self, gamma, mu=2):
		'''
		Returns a dictionary mapping categories to distributions
		created under the specified gamma and mu parameters. gamma may
		be set to 'uniform' to create uniform category distributions.
		'''
		if gamma == 'uniform':
			return {category:self.uniform_distribution(category) for category in self.iter_categories()}
		else:
			return {category:self.gaussian_distribution(category, gamma, mu) for category in self.iter_categories()}

	def uniform_distribution(self, category):
		'''
		Returns the uniform distribution for a particular category.
		'''
		category_members = self[category]
		uniform_probability = 1.0 / len(category_members)
		distribution = np.zeros(self.shape, dtype=float)
		for point in category_members:
			distribution[point] = uniform_probability
		return Distribution(distribution, normalize=False)

	def gaussian_distribution(self, category, gamma=1, mu=2):
		'''
		Returns the Gaussian distribution for a particular category
		under the specified gamma and mu parameters.
		'''
		distribution = np.zeros(self.shape, dtype=float)
		for point in self.iter_points():
			distribution[point] = self._category_similarity(point, category, gamma, mu)
		return Distribution(distribution, normalize=True)

	def _category_similarity(self, point, category, gamma, mu):
		'''
		Returns the sum similarity between a point and all members of
		a category under the specified gamma and mu parameters.
		'''
		return sum(self._similarity(point, member, gamma, mu) for member in self[category])

	def _similarity(self, x, y, gamma, mu):
		'''
		Returns the similarity between two points under the specified
		gamma and mu parameters.
		'''
		if not ((isinstance(gamma, int) or isinstance(gamma, float)) and gamma >= 0):
			raise ValueError('Gamma parameter must be positive number.')
		return np.exp(-gamma * self._distance(x, y, mu)**2)

	def _distance(self, x, y, mu):
		'''
		Returns the Minkowski distance between two points for some mu.
			mu = 1: Manhattan distance
			mu = 2: Euclidean distance
		'''
		if not ((isinstance(mu, int) or isinstance(mu, float)) and mu > 0):
			if mu == 'circle_euclidean':
				return self._circle_euclidean(x, y)
			raise ValueError('Mu parameter must be positive number.')
		return sum(abs(x - y)**mu for x, y in zip(x, y))**(1.0/mu)

	def _circle_euclidean(self, x, y):
		'''
		Returns the Euclidean distance between two points on a line
		which wraps back around on itself (the shorter distance in
		either direction is returned).
		'''
		sigma = 0.0
		for dim in range(len(self.shape)):
			d1 = abs(x[dim] - y[dim])
			d2 = abs(d1 - self.shape[dim])
			if d1 < d2:
				sigma += d1**2
			else:
				sigma += d2**2
		return sigma**0.5

########################################################################

class Distribution:

	'''
	A Distribution object represents a probability distribution. An
	error is raised if the passed probabilities do not sum to 1; to
	correct this, set normalize to True, which will automatically
	normalize the distribution.
	'''

	@property
	def shape(self):
		return self.probabilities.shape

	def __init__(self, distribution, normalize=False):
		distribution = np.array(distribution, dtype=float)
		if distribution.ndim == 0:
			raise ValueError('Distribution must have at least one dimension')
		if normalize is True:
			self.probabilities = distribution / distribution.sum()
		elif np.isclose(distribution.sum(), 1.0):
			self.probabilities = distribution
		else:
			raise ValueError('Probabilities do not sum to 1: Use normalize=True')

	def __repr__(self):
		'''
		Provides textual description of the distribution.
		'''
		dims = len(self.shape)
		start = '['*dims + str(self.probabilities[(0,)*dims])
		end = str(self.probabilities[(-1,)*dims]) + ']'*dims
		return 'Distribution%s ... %s' % (start, end)

	def __str__(self):
		'''
		Provides printable representation of the distribution.
		'''
		return self.probabilities.__str__()

	def __getitem__(self, key):
		'''
		Pass an int (1D) or tuple (ND) to get the probability of that
		point on the distribution.
		'''
		return self.probabilities[key]

	def __iter__(self):
		'''
		Default iterator. Each iteration returns a point in the
		distribution along with its associated probability.
		'''
		for point, probability in np.ndenumerate(self.probabilities):
			yield point, probability

	def __mul__(self, operand):
		return self.probabilities * operand.probabilities

	def smooth(self, alpha):
		'''
		Returns a smoothed copy of the Distribution using convex
		combination smoothing. alpha=0: no smoothing; alpha=1: smooth
		to a uniform distribution.
		'''
		if alpha:
			if not isinstance(alpha, (int, float)) and (alpha < 0 or alpha > 1):
				raise ValueError('Alpha must be number between 0 and 1.')
			uniform = np.full(self.shape, 1.0 / np.product(self.shape), dtype=float)
			return Distribution(uniform*alpha + self.probabilities*(1.0 - alpha), False)
		return self

########################################################################

class Need(Distribution):

	'''
	A Need object represents the probability with which each point in
	an n-dimensional space will need to be expressed. To create a Need
	object, pass a list like [[2,2,4,5], [3,1,6,8]], where the
	structure of the lists represents the space (here 2x4), and the
	numbers represent the frequency or probability of each point.
	Frequencies will automatically be converted to probabilities.
	Passing a tuple like (2,4) creates a Need object of given
	dimensionality with uniform need probabilities.
	'''

	def __init__(self, need_frequencies):
		if isinstance(need_frequencies, tuple):
			self.probabilities = np.full(need_frequencies, 1.0 / np.product(need_frequencies), dtype=float)
		else:
			need_frequencies = np.array(need_frequencies, dtype=float)
			if need_frequencies.ndim == 0:
				raise ValueError('Distribution must be at least one dimensional')
			self.probabilities = need_frequencies / need_frequencies.sum()

########################################################################

class Speaker:

	'''
	Collection of distributions - one for each point in the space.
	'''

	@property
	def shape(self):
		return self._shape

	def __init__(self, shape, speaker_distributions=None):
		if not isinstance(shape, tuple):
			raise ValueError('Shape must be tuple')
		self._shape = shape
		self._distributions = {}
		if speaker_distributions:
			if not isinstance(speaker_distributions, dict):
				raise ValueError('Speaker distributions shoud be passed as dictionary: point:distribution')
			else:
				points = list(np.ndindex(self._shape))
				for point in points:
					if point not in speaker_distributions:
						raise ValueError('Speaker distributions must be provided for every point')
				for point, speaker_distribution in speaker_distributions.items():
					if point not in points:
						raise ValueError('Invalid point contained in passed speaker distributions')
					self[point] = speaker_distribution
		else: # Assume speaker certainty and create point distributions
			for point in np.ndindex(self._shape):
				point_distribution = np.zeros(self._shape, dtype=float)
				point_distribution[point] = 1.0
				self._distributions[point] = Distribution(point_distribution)

	def __getitem__(self, key):
		'''
		Pass a tuple to get the category memebership of a point. Pass
		an integer to get a list of points that belong to a category.
		'''
		if key not in self._distributions:
			raise ValueError('Invalid point.')
		return self._distributions[key]

	def __setitem__(self, key, value):
		'''
		Change the category membership of a particular point.
		'''
		if not self._valid_key(key):
			raise ValueError('Invalid point.')
		if not isinstance(value, Distribution):
			value = Distribution(value)
		if value.shape != self._shape:
			raise ValueError('Distribution shape does not match the shape of the speaker.')
		self._distributions[key] = value

	def __iter__(self):
		'''
		Default iterator. Each iteration returns a point in the
		distribution along with its associated probability.
		'''
		for point in np.ndindex(self._shape):
			yield (point, self[point])

	def _valid_key(self, key):
		if not isinstance(key, tuple):
			return False
		if len(key) != len(self.shape):
			return False
		for dim in range(len(key)):
			if key[dim] >= self._shape[dim]:
				return False
		return True

########################################################################

class Listener:

	'''
	Collection of distributions - one for each category
	'''

	@property
	def shape(self):
		return self._shape

	def __init__(self, shape, listener_distributions):
		if not isinstance(shape, tuple):
			raise ValueError('Shape must be tuple')
		if not isinstance(listener_distributions, dict):
			raise ValueError('Listener distributions shoud be passed as dictionary: category:Distribution')
		self._shape = shape
		self._distributions = {}
		for category, listener_distribution in listener_distributions.items():
			self[category] = listener_distribution

	def __getitem__(self, key):
		'''
		Pass an integer to get the distribution for that category.
		'''
		if key not in self._distributions:
			raise ValueError('Invalid category.')
		return self._distributions[key]

	def __setitem__(self, key, value):
		'''
		Change the distribution for a particular category
		'''
		if not isinstance(value, Distribution):
			value = Distribution(value)
		if value.shape != self._shape:
			raise ValueError('Distribution shape does not match the shape of the listener.')
		self._distributions[key] = value

	def __iter__(self):
		'''
		Default iterator. Each iteration returns a point in the
		distribution along with its associated probability.
		'''
		for category in sorted(list(self._distributions.keys())):
			yield (category, self[category])

	def smooth(self, alpha):
		if alpha:
			smoothed_distributions = {}
			for category, distribution in self._distributions.items():
				smoothed_distributions[category] = distribution.smooth(alpha)
			return Listener(self.shape, smoothed_distributions)
		return self


########################################################################

class Space:

	'''
	A Space object represents an n-dimensional universe. To create a
	space object of certain dimensionality, pass a tuple like (2,4).
	Optionally, you can pass a need object specifying, a gamma setting
	(default: 1), a mu setting (default: 2 (Euclidean), 1 =
	Manhattan), If no need object is passed, a uniform need object
	will be created.
	'''

	@property
	def shape(self):
		return self._shape

	def __init__(self, shape, need=None, gamma=1, mu=2):
		if not isinstance(shape, tuple):
			raise ValueError('The shape of the space must be a tuple.')
		self._shape = shape
		if need:
			if not isinstance(need, Need):
				raise ValueError('Invalid need object. Pass a need object or set to None for uniform need probabilities.')
			self._need = need
		else: # Need unspecified, so create a uniform need object
			self._need = Need(self._shape)
		if not ((isinstance(gamma, int) or isinstance(gamma, float)) and gamma >= 0):
			raise ValueError('Gamma parameter must be positive number.')
		self._gamma = gamma
		if not ((isinstance(mu, int) or isinstance(mu, float)) and mu > 0):
			raise ValueError('Mu parameter must be positive number.')
		self._mu = mu
		pairwise_distances = pdist(list(np.ndindex(self._shape)), 'minkowski', self._mu)
		distance_matrix = squareform(pairwise_distances)
		self._similarity_matrix = np.exp(-self._gamma * distance_matrix**2)

	def __repr__(self):
		'''
		Provides textual description of the space object.
		'''
		if len(self._shape) == 1:
			return 'Space[length=%i, gamma=%i, mu=%s]' % (self._shape[0], self._gamma, self._mu)
		return 'Space[dimensionality=%s, gamma=%i, mu=%s]' % ('x'.join(map(str, self._shape)), self._gamma, self._mu)

	def communicative_cost(self, partition, need=None):
		'''
		Returns the communicative cost for a given partition and need
		probabilities. If no need object is passed, the need
		probabilities will be inherited from the space's own need
		object.
		'''
		if not isinstance(partition, Partition):
			raise ValueError('Invalid Partition object.')
		if partition.shape != self._shape:
			raise ValueError('Partition object does not match the dimensions of the space. Should be %s.' % 'x'.join(map(str, self._shape)))
		if need:
			if not isinstance(need, Need):
				raise ValueError('Invalid Need object. Pass a Need object or set to None to inherit need probabilities from the space.')
			if need.shape != self._shape:
				raise ValueError('Need object does not match the dimensions of the space. Should be %s.' % 'x'.join(map(str, self._shape)))
		else:
			need = self._need
		boolean_matrix = partition.boolean_matrix()
		listener_distributions = np.dot(self._similarity_matrix, boolean_matrix)
		norm_listener_distributions = listener_distributions * boolean_matrix / listener_distributions.sum(axis=0)
		neg_log_listener_distributions = -np.log2(norm_listener_distributions.sum(axis=1))
		return (need.probabilities * neg_log_listener_distributions.reshape(self._shape)).sum()

	def cost(self, language_array):
		'''
		Returns the communicative cost of a language passed as a
		simple numpy array under the assumption of uniform need
		probabilities. Essentially does the same as the
		communicative_cost method above without the need to first
		convert the numpy array to a Partition object.
		'''
		if not isinstance(language_array, np.ndarray):
			raise ValueError('language_array should be Numpy array')
		if language_array.shape != self._shape:
			raise ValueError('Partition object does not match the dimensions of the space. Should be %s.' % 'x'.join(map(str, self._shape)))
		boolean_matrix = convert_to_bool_matrix(language_array)
		listener_distributions = np.dot(self._similarity_matrix, boolean_matrix)
		norm_listener_distributions = listener_distributions * boolean_matrix / listener_distributions.sum(axis=0)
		neg_log_listener_distributions = -np.log2(norm_listener_distributions.sum(axis=1))
		return (self._need.probabilities * neg_log_listener_distributions.reshape(self._shape)).sum()

########################################################################

def convert_to_bool_matrix(partition):
	'''
	Returns a 2D Boolean matrix, where rows correspond to meanings and
	columns correspond to categories. True indicates that the ith
	meaning belongs to the jth category. This Boolean matrix
	representation is used by the communicative_cost method in the
	Space object for fast computation using a similarity matrix.
	'''
	n_points = partition.size # determines number of rows
	n_categories = len(np.unique(partition)) # determines number of columns
	cat_to_col = {cat:col for col, cat in enumerate(np.unique(partition))} # maps categories to columns
	boolean_matrix = np.zeros((n_points, n_categories), dtype=bool)
	for row, point in enumerate(np.ndindex(partition.shape)):
		column = cat_to_col[partition[point]]
		boolean_matrix[row, column] = True
	return boolean_matrix

########################################################################

def KL_divergence(s, l):
	'''
	Returns the KL divergence between a speaker and listener
	distribution.
	'''
	if s.shape != l.shape:
		raise ValueError('Speaker and listener distributions do not have the same shape')
	D_KL = 0.0
	for point in np.ndindex(s.shape):
		if s[point] == 0:
			continue
		if l[point] == 0:
			raise ValueError('Cannot compute KL divergence because l=0 where s>0 at point %s. Try smoothing.'%str(point))
		D_KL += s[point] * np.log2(s[point] / (l[point]))
	return D_KL

def cost(partition, need, speaker, listener, alpha=None):
	'''
	Returns the communicative cost given partition, need, speaker, and
	listener objects.
	'''
	if not isinstance(partition, Partition):
		raise ValueError('Invalid Partition object')
	if not isinstance(need, Need) or partition.shape != need.shape:
		raise ValueError('Invalid Need object')
	if not isinstance(speaker, Speaker) or partition.shape != speaker.shape:
		raise ValueError('Invalid Speaker object')
	if not isinstance(listener, Listener) or partition.shape != listener.shape:
		raise ValueError('Invalid Listener object')
	if alpha:
		listener = listener.smooth(alpha)
	return sum(need[target] * KL_divergence(speaker[target], listener[category]) for target, category in partition)

########################################################################

def random_partition(shape, n_categories, convex=False, seeds=None):
	'''
	Returns a randomly generated partition object with specified
	shape, number of categories, and convexity.
	'''
	space = np.full(shape, -1, dtype=int)
	n_items = np.product(shape)
	points = list(np.ndindex(shape))
	if seeds is None:
		seeds = [points[p] for p in np.random.choice(n_items, n_categories, False)]
	for category in range(n_categories):
		space[seeds[category]] = category
	for point in points:
		if space[point] == -1:
			if convex:
				distances = [dist(point, seed, 2) for seed in seeds]
				min_distance = min(distances)
				category = np.random.choice([c for c in range(n_categories) if distances[c] == min_distance])
			else:
				category = np.random.choice(n_categories)
			space[point] = category
	return seeds, space

def iter_partitions(collection):
	if len(collection) == 1:
		yield [ collection ]
		return
	first = collection[0]
	for smaller in iter_partitions(collection[1:]):
		for n, subset in enumerate(smaller):
			yield smaller[:n] + [[ first ] + subset] + smaller[n+1:]
		yield [ [ first ] ] + smaller

def all_partitions(shape):
	'''
	Returns all partitions of a space
	'''
	space = np.zeros(shape, dtype=int)
	for partition in iter_partitions(list(np.ndindex(shape))):
		for category, points in enumerate(partition):
			for point in points:
				space[point] = category
		yield Partition(space)

def dist(x, y, mu):
	return sum(abs(x - y)**mu for x, y in zip(x, y))**(1.0/mu)
