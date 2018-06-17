'''
Bayesian iterated learning model of the emergence of conceptual
structure under simplicity or informativeness priors. There are two
classes:

	Agent : represents a Bayesian agent
	Chain : represents an iterated learning chain

To run a simulation, create a Chain object with desired parameters:

	chain = Chain(
		generations = 10,
		maxcats = 4,
		prior = 'simplicity',
		bottleneck = 2,
		exposures = 4,
		noise = 0.05
	)

and then call the simulate method:

	chain.simulate('results.json')

Results are written out to the specified file. Each line contains
data for one generation. Initially, a random language is generated
for the first agent to learn from. Then, the data output from each
subsequent agent becomes the input to the following agent in the
chain.
'''

import numpy as np
import rectlang
import commcost
import varofinf

# rectlang settings
rotational_invariance = True
max_exhaustive_size = 16
max_beam_width = 2

# commcost settings
gamma = 1
mu = 2

class Agent:

	'''
	An Agent object represents a Bayesian, category-learning agent.
	The agent is set up with some parameters and can then learn from
	data and produce signals for meanings. All probability
	calculations are done in the log domain (base 2). An agent is
	created with desired parameters like this:

		agent = Agent(maxcats=4, prior='simplicity', etc...)

	A dataset is list of meaning/signal pairs:

		data = [((0,0), 0), ((1,0), 0),
		        ((0,2), 1), ((1,2), 1),
		        ((2,1), 2), ((3,1), 2),
		        ((2,3), 3), ((3,3), 3)]
	
	The agent learns from the data like this:
		agent.learn(data)

	The agent's inferred language is accessed like this:
		agent.language

	The agent produces a signal for a given meaning like this:
		agent.speak( (3,2) )

	The agent produces signals for all meanings like this:
		agent.speak_all()
	'''

	def __init__(self, shape=(8,8), maxcats=4, prior='simplicity', weight=1.0, noise=0.05, exposures=4, mcmc_iterations=5000):

		if not isinstance(shape, tuple) or len(shape) != 2:
			raise ValueError('shape should be a tuple containing the height and width')
		if not isinstance(shape[0], int) or shape[0] < 1:
			raise ValueError('height must be positive integer')
		if not isinstance(shape[1], int) or shape[1] < 1:
			raise ValueError('width must be postive integer')
		self._shape = shape
		self._size = np.product(shape)
		self._cells, self._rects = self._generate_cells_and_rects()

		if not isinstance(maxcats, int) or maxcats < 1 or maxcats > self._size:
			raise ValueError('Invalid maxcats: Should be int between 1 and %i' % self._size)
		self._maxcats = maxcats

		if prior == 'simplicity':
			self._prior_grid = rectlang.Space(self._shape, rotational_invariance, max_exhaustive_size, max_beam_width)
			self._prior_func = self._prior_grid.complexity
		elif prior == 'informativeness':
			self._prior_grid = commcost.Space(self._shape, gamma=gamma, mu=mu)
			self._prior_func = self._prior_grid.cost
		elif not callable(prior):
			raise ValueError('Invalid prior: Use \'simplicity\' or \'informativeness\', or pass a callable function')
		else:
			self._prior_func = prior	

		if not isinstance(weight, (float, int)):
			raise ValueError('Invalid weight: Should be float or int')
		self._weight = weight

		if not isinstance(noise, float) or noise < 0 or noise > 1:
			raise ValueError('Invalid noise: Should be float between 0 and 1')
		self._noise = noise

		if not isinstance(exposures, int) or exposures < 1:
			raise ValueError('Invalid exposures: Should be int greater than 0')
		self._exposures = exposures

		if not isinstance(mcmc_iterations, int) or mcmc_iterations < 1000:
			raise ValueError('Invalid mcmc_iterations: Should be int greater than or equal to 1000')
		self._mcmc_iterations = mcmc_iterations

		self._prob_correct = np.log2(1.0 - self._noise)
		self._prob_incorrect = np.log2(self._noise / (self._maxcats-1))

		self.language = None

	def _generate_cells_and_rects(self):
		'''
		Generates a list of all rectangles (including all 1x1 cells)
		that will fit inside the space. This is performed once during
		initialization and the rectangles are then used by the
		propose_candidate function to select a region of the space to
		mutate.
		'''
		cells, rects = [], []
		for h in range(1, self._shape[0]+1): # height
			for w in range(1, self._shape[1]+1): # width
				for y in range(self._shape[0]+1-h): # y-coordinate
					for x in range(self._shape[1]+1-w): # x-coordinate
						if h == 1 and w == 1:
							cells.append((h, w, y, x))
						else:
							rects.append((h, w, y, x))
		return cells, rects

	def _posterior(self, language, data):
		'''
		Calculate posterior(L|D) = likelihood(D|L)^x * prior(L)^w, 
		where w is the prior weight and x is the number of exposures
		(rather than expose the agent to the dataset multiple times,
		the likelihood is simply raised to the number of exposures).
		'''
		likelihood, prior = 0.0, 0.0
		if self._exposures > 0.0:
			likelihood = self._likelihood(data, language) * self._exposures
		if self._weight > 0.0:
			prior = self._prior(language) * self._weight
		return likelihood + prior

	def _prior(self, language):
		'''
		Calculate prior(L) = 2^-f(L), where f is the prior function,
		a function that takes a language (partition of the space) as
		input and returns some measure of interest in bits (typically
		the complexity or communicative cost of the language).
		'''
		return -self._prior_func(language)

	def _likelihood(self, data, language):
		'''
		Calculate likelihood(D|L) = product[over <m,s> in D] p(s|L,m)
		'''
		likelihood = 0.0
		for meaning, signal in data:
			likelihood += self._signal_probability(signal, language, meaning)
		return likelihood

	def _signal_probability(self, signal, language, meaning):
		'''
		Calculate p(s|L,m) = 1 - ε        if L(m) == s
		                   = ε / (Nmax-1) if L(m) != s
		'''
		if language[meaning] == signal:
			return self._prob_correct
		return self._prob_incorrect

	def _get_mutables(self, language):
		'''
		Returns a list of rectangular areas in the language in which
		only one signal is used. This includes all individual 1x1
		cells, which, by definition, only contain one signal. This is
		used to propose a candidate language in the sampling process.
		'''
		mutable_rects = [(h, w, y, x) for h, w, y, x in self._rects if all_same(language[y:y+h, x:x+w])]
		return mutable_rects + self._cells

	def _propose_candidate(self, language, mutables):
		'''
		Given a language and list of mutable areas (rectangular areas
		in the language that all belong to one category), select one
		of the areas at random and change all meanings inside it to
		some other signal, creating a new candidate language.
		'''
		h, w, y, x = mutables[np.random.randint(len(mutables))]
		random_signal = np.random.randint(self._maxcats-1)
		if random_signal >= language[y,x]:
			random_signal += 1
		cand_language = language.copy()
		cand_language[y:y+h, x:x+w] = random_signal
		return cand_language

	##################
	# PUBLIC METHODS #
	##################

	def learn(self, data, language=None):
		'''
		Given some data, sample a language from the posterior using
		the Metropolis-Hastings algorithm. Measure the posterior of
		the agent's (initially random) language, then repeat the
		following lots of times: Propose a candidate language and
		measure the posterior again; calculate the acceptance ratio
		alpha = posterior ratio * proposal ratio; if alpha > 1, accept
		the candidate automatically; if alpha < 1, accept the
		candidate with probability alpha. Ultimately, this yields a
		representative sample from the true posterior. Optionally, a
		language can be passed in as the agent's starting hypothesis.
		Under the proposal function, there is exactly one mutable that
		allows you to jump from the language to the candidate, and
		exactly one mutable that allows you to jump back again, so the
		proposal probabilities are calculated as 1 / number of
		mutables (which changes slightly depending on the direction,
		making the function asymmetric).
		'''
		if language is None:
			language = np.random.randint(0, self._maxcats, self._shape)
		mutables = self._get_mutables(language)
		posterior = self._posterior(language, data)
		for _ in range(self._mcmc_iterations):
			cand_language = self._propose_candidate(language, mutables)
			cand_mutables = self._get_mutables(cand_language)
			cand_posterior = self._posterior(cand_language, data)
			prob_c_given_l = -np.log2(len(mutables)) # p(cand|lang)
			prob_l_given_c = -np.log2(len(cand_mutables)) # p(lang|cand)
			alpha = (cand_posterior - posterior) + (prob_l_given_c - prob_c_given_l)
			if (alpha >= 0.0) or (np.log2(np.random.random()) < alpha):
				language = cand_language
				mutables = cand_mutables
				posterior = cand_posterior
		self.language = language

	def speak(self, meaning):
		'''
		Returns a signal for a given meaning with some chance of
		error according to the noise parameter. Uses a roulette wheel
		to select a signal.
		'''
		random_prob = np.log2(np.random.random())
		summation = self._signal_probability(0, self.language, meaning)
		for signal in range(1, self._maxcats):
			if random_prob < summation:
				return signal - 1
			signal_prob = self._signal_probability(signal, self.language, meaning)
			summation = np.logaddexp2(summation, signal_prob)
		return signal

	def speak_all(self):
		'''
		Returns a 2D array of signals (productions) for every meaning
		in the space with some chance of error according to the noise
		parameter.
		'''
		productions = np.zeros(self._shape, dtype=int)
		for meaning in np.ndindex(self._shape):
			productions[meaning] = self.speak(meaning)
		return productions


class Chain:

	'''
	A Chain object simulates an iterated learning chain. The chain is
	set up with a set of parameters and is then run by calling the
	simulate() method:

		chain = Chain(generations=10, maxcats=4, etc...)
		chain.simulate('results.json')
	'''

	def __init__(self, generations=10, shape=(8,8), mincats=1, maxcats=4, prior='simplicity', weight=1.0, noise=0.05, bottleneck=2, exposures=4, mcmc_iterations=5000):
		if not isinstance(generations, int) or generations < 1:
			raise ValueError('Invalid generations: Should be int greater than 0')
		self._generations = generations

		if not isinstance(shape, tuple) or len(shape) != 2:
			raise ValueError('shape should be a tuple containing the height and width')
		if not isinstance(shape[0], int) or shape[0] < 1:
			raise ValueError('height must be positive integer')
		if not isinstance(shape[1], int) or shape[1] < 1:
			raise ValueError('width must be postive integer')
		self._shape = shape
		self._size = np.product(shape)

		if not isinstance(mincats, int) or mincats < 1 or mincats > self._size:
			raise ValueError('Invalid mincats: Should be int between 1 and %i' % self._size)
		self._mincats = mincats

		if not isinstance(maxcats, int) or maxcats < self._mincats or maxcats > self._size:
			raise ValueError('Invalid maxcats: Should be int between %i and %i' % (self._mincats, self._size))
		self._maxcats = maxcats

		self._rectlang_grid = rectlang.Space(self._shape, rotational_invariance, max_exhaustive_size, max_beam_width)
		self._commcost_grid = commcost.Space(self._shape, gamma=gamma, mu=mu)
		if prior == 'simplicity':
			self._prior_func = self._rectlang_grid.complexity
		elif prior == 'informativeness':
			self._prior_func = self._commcost_grid.cost
		elif not callable(prior):
			raise ValueError('Invalid prior: Use \'simplicity\' or \'informativeness\', or pass a callable function')
		else:
			self._prior_func = prior
		self._prior = str(prior)

		if not isinstance(weight, (float, int)):
			raise ValueError('Invalid weight: Must be float or int')
		self._weight = weight

		if not isinstance(noise, float) or noise < 0 or noise > 1:
			raise ValueError('Invalid noise: Must be float between 0 and 1')
		self._noise = noise

		if not isinstance(bottleneck, int) or bottleneck < 1 or bottleneck > 4:
			raise ValueError('Invalid bottleneck: Should be int between 1 and 4')
		self._bottleneck = bottleneck
		self._segments = self._segment_space()

		if not isinstance(exposures, int) or exposures < 1:
			raise ValueError('Invalid exposures: Must be int greater than 0')
		self._exposures = exposures

		if not isinstance(mcmc_iterations, int) or mcmc_iterations < 1000:
			raise ValueError('Invalid mcmc_iterations: Should be int greater than or equal to 1000')
		self._mcmc_iterations = mcmc_iterations

		self._model_parameters = {'shape':self._shape, 'mincats':self._mincats, 'maxcats':self._maxcats, 'prior':self._prior, 'weight':self._weight, 'noise':self._noise, 'bottleneck':self._bottleneck, 'exposures':self._exposures, 'mcmc_iterations':self._mcmc_iterations}

	def _segment_space(self):
		'''
		Break the space up into 2x2 segments in order to sample
		meanings evenly over the space. b meanings will be sampled
		from each segment, where b is the bottleneck parameter.
		'''
		segments = []
		for y in range(0, self._shape[0], 2):
			for x in range(0, self._shape[1], 2):
				segments.append([(y,x), (y,x+1), (y+1,x), (y+1,x+1)])
		return segments

	def _initial_language(self):
		'''
		Generate a random language with an equal (or close to equal)
		number of points in each category. If the size of the space
		is a multiple of maxcats, there will be an equal number of
		points in each category. This is used to create the initial
		generation-0 language.
		'''
		language = []
		for signal in range(self._maxcats):
			language.extend([signal] * (self._size // self._maxcats))
		n_extras = self._size - len(language)
		language.extend(np.random.choice(self._maxcats, n_extras, False))
		language = np.array(language, dtype=int)
		np.random.shuffle(language)
		return language.reshape(self._shape)

	def _new_agent(self, language=None):
		'''
		Create a new agent, inheriting the chain's parameters. If a
		language is passed, the agent is initialized with that
		language.
		'''
		agent = Agent(self._shape, self._maxcats, self._prior_func, self._weight, self._noise, self._exposures, self._mcmc_iterations)
		if language is not None:
			if not isinstance(language, np.ndarray) or language.shape != self._shape or language.max() >= self._maxcats:
				raise ValueError('Invalid language: Cannot override agent\'s language')
			agent.language = language
		return agent

	def _generate_data(self, productions):
		'''
		From each of the 2x2 segments, select b meanings at random,
		where b is the bottleneck parameter. These meanings are then
		paired with their corresponding signals to form a dataset.
		'''
		data = []
		for segment in self._segments:
			for selection in np.random.choice(4, self._bottleneck, replace=False):
				meaning = segment[selection]
				data.append((meaning, productions[meaning]))
		return data

	def _write_generation(self, output_file, language, productions, data, filtered=False, lang_vi=None, prod_vi=None):
		'''
		Write generation data to a given file, including measurements
		of expressivity, transmission error, complexity, and
		communicative cost. Each generation is appended to the file.
		'''
		generation_data = {
			'language':language.flatten().tolist(),
			'productions':productions.flatten().tolist(),
			'data_out':data,
			'filtered_agent':filtered,
			'lang_expressivity':len(np.unique(language)),
			'prod_expressivity':len(np.unique(productions)),
			'lang_error':lang_vi,
			'prod_error':prod_vi,
			'lang_complexity':self._rectlang_grid.complexity(language),
			'prod_complexity':self._rectlang_grid.complexity(productions),
			'lang_cost':self._commcost_grid.cost(language),
			'prod_cost':self._commcost_grid.cost(productions),
			'model_parameters':self._model_parameters
		}
		with open(output_file, mode='a') as file:
			file.write(str(generation_data) + '\n')

	def _begin_chain(self, output_file):
		'''
		Set up a new chain by creating an initial generation-0
		language, a dummy generation-0 agent, productions, and an
		initial dataset that generation 1 will be trained on.
		'''
		language = self._initial_language()
		agent = self._new_agent(language)
		productions = agent.speak_all()
		data = self._generate_data(productions)
		self._write_generation(output_file, language, productions, data)
		return 1, agent, productions, data

	def _resume_chain(self, output_file):
		'''
		Open a previously started chain output file and load in the
		last generation in order to resume running the chain.
		'''
		generation_i = 0
		with open(output_file, mode='r') as file:
			for line in file:
				line = line.strip()
				if line == '': break
				generation = eval(line)
				if not generation['filtered_agent']:
					last_generation = generation
					generation_i += 1
		last_language = np.array(last_generation['language'], dtype=int).reshape(self._shape)
		last_productions = np.array(last_generation['productions'], dtype=int).reshape(self._shape)
		last_data_out = last_generation['data_out']
		agent = self._new_agent(last_language)
		return generation_i, agent, last_productions, last_data_out

	##################
	# PUBLIC METHODS #
	##################

	def simulate(self, output_file, resume=False):
		'''
		Runs an iterated learning chain. For each generation in the
		chain, an agent is created that learns from the previous
		agent in the chain. The inferred language and output data is
		written to a data file at each generation. If mincats is less
		than maxcats and an agent infers a language with fewer than
		mincats categories, that agent is removed, new data is
		generated from the previous generation, and a new agent is
		given the data to learn. If resume is set to True, the
		program attempts to resume running the chain from the file.
		'''
		if resume:
			generation_i, agent, last_productions, data = self._resume_chain(output_file)
		else:
			generation_i, agent, last_productions, data = self._begin_chain(output_file)
		while generation_i <= self._generations:
			new_agent = self._new_agent()
			new_agent.learn(data)
			productions = new_agent.speak_all()
			lang_vi = varofinf.variation_of_information(agent.language, new_agent.language)
			prod_vi = varofinf.variation_of_information(last_productions, productions)
			if len(np.unique(agent.language)) >= self._mincats:
				data = self._generate_data(productions)
				self._write_generation(output_file, new_agent.language, productions, data, False, lang_vi, prod_vi)
				agent = new_agent
				last_productions = productions
				generation_i += 1
			else: # revert to previous agent and generate new data
				productions = agent.speak_all()
				data = self._generate_data(productions)
				self._write_generation(output_file, new_agent.language, productions, data, True, lang_vi, prod_vi)


def all_same(array):
	'''
	Returns True if all values in the array are equal.
	'''
	array = array.flatten()
	last_value = array[-1]
	for value in array:
		if value != last_value:
			return False
	return True


if __name__ == '__main__':

	import os
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('output_path', action='store', type=str, help='directory to write data to')
	parser.add_argument('chain_i', action='store', type=int, help='chain number')
	parser.add_argument('--generations', action='store', type=int, default=100, help='number of generations in a chain (100)')
	parser.add_argument('--height', action='store', type=int, default=8, help='height of the meaning space')
	parser.add_argument('--width', action='store', type=int, default=8, help='width of the meaning space')
	parser.add_argument('--mincats', action='store', type=int, default=1, help='minimum number of categories an agent must infer to be iterated (1)')
	parser.add_argument('--maxcats', action='store', type=int, default=4, help='maximum number of categories an agent can infer (4)')
	parser.add_argument('--prior', action='store', type=str, default='simplicity', help='type of prior to use (\'simplicity\')')
	parser.add_argument('--weight', action='store', type=float, default=1.0, help='weighting of the prior (1)')
	parser.add_argument('--noise', action='store', type=float, default=0.05, help='probability of noise on production (0.05)')
	parser.add_argument('--bottleneck', action='store', type=int, default=2, help='number of transmitted meanings per 2x2 segment (2)')
	parser.add_argument('--exposures', action='store', type=int, default=4, help='number of exposures to the training set (4)')
	parser.add_argument('--mcmc_iterations', action='store', type=int, default=5000, help='number of MCMC iterations (5000)')
	args = parser.parse_args()

	try:
		os.makedirs(args.output_path)
	except FileExistsError:
		pass

	chain = Chain(args.generations, (args.height, args.width), args.mincats, args.maxcats, args.prior, args.weight, args.noise, args.bottleneck, args.exposures, args.mcmc_iterations)
	output_file = os.path.join(args.output_path, str(chain_i))
	chain.simulate(output_file, os.path.isfile(output_file))
