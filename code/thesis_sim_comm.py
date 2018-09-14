'''
Code for simulating communication between participants. This was not
included in the paper; refer to my thesis instead.
'''

import matplotlib.pyplot as plt
import numpy as np
import tools
import colors
import pickle

plt.rcParams['svg.fonttype'] = 'none'

partitions = {
	'angle': {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1,26:1,27:1,28:1,29:1,30:1,31:1,32:2,33:2,34:2,35:2,36:2,37:2,38:2,39:2,40:2,41:2,42:2,43:2,44:2,45:2,46:2,47:2,48:3,49:3,50:3,51:3,52:3,53:3,54:3,55:3,56:3,57:3,58:3,59:3,60:3,61:3,62:3,63:3},
	'size' : {0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3,8:0,9:0,10:1,11:1,12:2,13:2,14:3,15:3,16:0,17:0,18:1,19:1,20:2,21:2,22:3,23:3,24:0,25:0,26:1,27:1,28:2,29:2,30:3,31:3,32:0,33:0,34:1,35:1,36:2,37:2,38:3,39:3,40:0,41:0,42:1,43:1,44:2,45:2,46:3,47:3,48:0,49:0,50:1,51:1,52:2,53:2,54:3,55:3,56:0,57:0,58:1,59:1,60:2,61:2,62:3,63:3},
	'both' : {0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:1,8:0,9:0,10:0,11:0,12:1,13:1,14:1,15:1,16:0,17:0,18:0,19:0,20:1,21:1,22:1,23:1,24:0,25:0,26:0,27:0,28:1,29:1,30:1,31:1,32:2,33:2,34:2,35:2,36:3,37:3,38:3,39:3,40:2,41:2,42:2,43:2,44:3,45:3,46:3,47:3,48:2,49:2,50:2,51:2,52:3,53:3,54:3,55:3,56:2,57:2,58:2,59:2,60:3,61:3,62:3,63:3}
}

def extract_speakers(dataset):
	speakers_by_condition = {'angle':[], 'size':[], 'both':[]}
	for participant in dataset:
		if participant['test_type'] == 'production' and participant['status'] == 'finished':
			language = np.zeros(64, dtype=int)
			for answer, response in zip(participant['test_sequence'], participant['test_responses']):
				language[answer] = response
			language = language.reshape((8,8))
			speakers_by_condition[participant['condition']].append(language)
	return speakers_by_condition

def extract_listeners(dataset):
	listeners_by_condition = {'angle':[], 'size':[], 'both':[]}
	for participant in dataset:
		if participant['test_type'] == 'comprehension' and participant['status'] == 'finished':
			language = {0:[], 1:[], 2:[], 3:[]}
			for answer, response in zip(participant['test_sequence'], participant['test_responses']):
				meaning = response // 8, response % 8
				language[answer].append(meaning)
			listeners_by_condition[participant['condition']].append(language)
	return listeners_by_condition

def distance(meaning1, meaning2):
	d = ((meaning1[0]-meaning2[0]))**2 + ((meaning1[1]-meaning2[1]))**2
	return np.sqrt(d)

def simulate_similarity_accuracy(speakers, listeners, n_sims=100):
	scores = {'angle':[], 'size':[], 'both':[]}
	for condition in scores.keys():
		for _ in range(n_sims):
			speaker = speakers[condition][np.random.randint(40)]
			listener = listeners[condition][np.random.randint(40)]
			target = np.random.randint(8), np.random.randint(8)
			signal = speaker[target]
			meanings = listener[signal]
			target_prime = meanings[np.random.randint(len(meanings))]
			scores[condition].append(distance(target, target_prime))
	return scores

def simulate_binary_accuracy(speakers, listeners, n_sims=100):
	scores = {'angle':[], 'size':[], 'both':[]}
	for condition in scores.keys():
		for speaker in speakers[condition]:
			for listener in listeners[condition]:
				score = 0
				for _ in range(n_sims):
					for meaning in np.ndindex((8,8)):
						signal = speaker[meaning]
						inferred_meaning = listener[signal][np.random.randint(len(listener[signal]))]
						if meaning == inferred_meaning:
							score += 1
				score /= (n_sims*64)
				scores[condition].append(score)
	return scores

def plot_densities(axis, results):
	positions = [0.6, 0.3, 0]
	y_lim = [0, 0.9]
	distributions = [results['angle'], results['size'], results['both']]
	violins = axis.violinplot(distributions, positions, vert=False, showmedians=True, showextrema=False)
	for body in violins['bodies']:
		m = np.mean(body.get_paths()[0].vertices[:, 1])
		body.get_paths()[0].vertices[:, 1] = np.clip(body.get_paths()[0].vertices[:, 1], m, np.inf)
		body.set_facecolor(colors.black)
		body.set_edgecolor(colors.black)
		body.set_alpha(1.0)
	violins['cmedians'].set_color('white')
	# axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
	# axis.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
	axis.set_yticklabels([])
	axis.tick_params(axis='y', which='both', left='off', right='off')
	axis.set_ylim(*y_lim)

def plot_comm_accuracy(production_results, figure_path):
	fig, axis = plt.subplots(1, 1, figsize=(1.8, 2))
	plot_densities(axis, production_results)
	axis.set_xlabel('Communicative error', fontsize=8)
	axis.set_xlim(0,5)
	for tick in axis.xaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5)
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)



def plot_comm_success(participant_results, simulated_results, figure_path):
	fig, axes = plt.subplots(1, 2, figsize=(4.5, 2))
	plot_densities(axes[0], participant_results)
	plot_densities(axes[1], simulated_results)
	axes[0].set_title('Simulated participant interactions', fontsize=8)
	axes[1].set_title('Simulated agent interactions', fontsize=8)
	axes[0].set_xlabel('Communicative error', fontsize=8)
	axes[1].set_xlabel('Communicative error', fontsize=8)
	axes[0].set_xlim(0,10)
	axes[1].set_xlim(0,10)
	for tick in axes[0].xaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axes[1].xaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5)
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)



class listener:

	def __init__(self, partition, gamma=1.0):
		distributions = { 0:np.zeros(partition.shape,dtype=float), 1:np.zeros(partition.shape,dtype=float), 2:np.zeros(partition.shape,dtype=float), 3:np.zeros(partition.shape,dtype=float) }
		for signal in distributions.keys():
			for meaning in np.ndindex(partition.shape):
				for meaning_, signal_ in np.ndenumerate(partition):
					if signal_ == signal:
						distributions[signal][meaning] += np.exp(-gamma * distance(meaning, meaning_)**2)
			distributions[signal] = np.log2(distributions[signal] / distributions[signal].sum()).reshape(partition.size)
		self.distributions = distributions
		self.size = partition.size

	def infer(self, signal):
		random_prob = np.log2(np.random.random())
		summation = self.distributions[signal][0]
		for meaning in range(1, self.size):
			if random_prob < summation:
				return meaning - 1
			meaning_prob = self.distributions[signal][meaning]
			summation = np.logaddexp2(summation, meaning_prob)
		return meaning



def save_results(array, filename):
	with open(filename, mode='wb') as file:
		pickle.dump(array, file)

def restore_results(filename):
	with open(filename, mode='rb') as file:
		array = pickle.load(file)
	return array


def make_stripes(width):
	a = np.zeros((width, width), dtype=int)
	z = width // 4
	for i in range(4):
		a[i*z:(i*z)+z, :] = i
	return a

def make_quads(width):
	a = np.zeros((width, width), dtype=int)
	z = width // 2
	a[0:z, z:] = 1
	a[z:, 0:z] = 2
	a[z:, z:] = 3
	return a

def simulate(system, sims=10000):
	l = listener(system)
	scores = []
	for _ in range(sims):
		target = np.random.randint(32), np.random.randint(32)
		signal = system[target]
		target_prime = l.infer(signal)
		target_prime = target_prime // 32, target_prime % 32
		scores.append(distance(target, target_prime))
	return scores

def run(results_file):
	S = make_stripes(32)
	Q = make_quads(32)
	scores = {'angle':simulate(S), 'size':simulate(S), 'both':simulate(Q)}
	save_results(scores, results_file)

def plot(scores):
	plot_comm_accuracy(scores, '/Users/jon/Desktop/accuracy.svg')

simulated_results = restore_results('/Users/jon/Desktop/sim_scores')
dataset = tools.read_json_lines('../data/experiments/exp1_participants.json')
speakers_by_condition = extract_speakers(dataset)
listeners_by_condition = extract_listeners(dataset)


scores_dis = simulate_similarity_accuracy(speakers_by_condition, listeners_by_condition, 1000)

plot_comm_success(scores_dis, simulated_results, '/Users/jon/Desktop/accuracy.svg')


# run('/Users/jon/Desktop/sim_scores')
# plot(restore_results('/Users/jon/Desktop/sim_scores'))

