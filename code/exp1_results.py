import numpy as np
import matplotlib.pyplot as plt
import visualize
import tools

plt.rcParams['svg.fonttype'] = 'none'

partitions = {
	'angle': {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1,26:1,27:1,28:1,29:1,30:1,31:1,32:2,33:2,34:2,35:2,36:2,37:2,38:2,39:2,40:2,41:2,42:2,43:2,44:2,45:2,46:2,47:2,48:3,49:3,50:3,51:3,52:3,53:3,54:3,55:3,56:3,57:3,58:3,59:3,60:3,61:3,62:3,63:3},
	'size' : {0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3,8:0,9:0,10:1,11:1,12:2,13:2,14:3,15:3,16:0,17:0,18:1,19:1,20:2,21:2,22:3,23:3,24:0,25:0,26:1,27:1,28:2,29:2,30:3,31:3,32:0,33:0,34:1,35:1,36:2,37:2,38:3,39:3,40:0,41:0,42:1,43:1,44:2,45:2,46:3,47:3,48:0,49:0,50:1,51:1,52:2,53:2,54:3,55:3,56:0,57:0,58:1,59:1,60:2,61:2,62:3,63:3},
	'both' : {0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:1,8:0,9:0,10:0,11:0,12:1,13:1,14:1,15:1,16:0,17:0,18:0,19:0,20:1,21:1,22:1,23:1,24:0,25:0,26:0,27:0,28:1,29:1,30:1,31:1,32:2,33:2,34:2,35:2,36:3,37:3,38:3,39:3,40:2,41:2,42:2,43:2,44:3,45:3,46:3,47:3,48:2,49:2,50:2,51:2,52:3,53:3,54:3,55:3,56:2,57:2,58:2,59:2,60:3,61:3,62:3,63:3}
}

def plot_densities(axis, results):
	positions = [0, -0.4, -0.8]
	axis.plot([0.25, 0.25], [-0.8, 0.3], c='black', linestyle='--', zorder=0)
	violins = axis.violinplot([results['angle'], results['size'], results['both']], positions, vert=False, showmedians=True, showextrema=False)
	for i, body in enumerate(violins['bodies']):
		m = np.mean(body.get_paths()[0].vertices[:, 1])
		body.get_paths()[0].vertices[:, 1] = np.clip(body.get_paths()[0].vertices[:, 1], m, np.inf)
		body.set_facecolor('#323536')
		body.set_edgecolor('#323536')
		body.set_alpha(1.0)
	violins['cmedians'].set_color('white')
	axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
	axis.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
	axis.set_yticklabels([])
	axis.tick_params(axis='y', which='both', left='off', right='off')
	axis.set_xlabel('Proportion correct')
	axis.set_xlim(0,1)
	axis.set_ylim(-0.8, 0.3)

def plot_prop_correct(production_results, comprehension_results, file_path):
	fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
	plot_densities(axes[0], production_results)
	plot_densities(axes[1], comprehension_results)
	axes[0].set_title('Production', fontsize=10)
	axes[1].set_title('Comprehension', fontsize=10)
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5)
	fig.savefig(file_path, format='svg')
	tools.format_svg_labels(file_path)
	if not file_path.endswith('.svg'):
		tools.convert_svg(file_path, file_path)

def mean_prop_correct_production(dataset):
	results = { 'angle':[], 'size':[], 'both':[] }
	for participant in dataset:
		if participant['status'] == 'finished':
			score = 0.0
			for answer, response in zip(participant['test_sequence'], participant['test_responses']):
				if partitions[participant['condition']][answer] == response:
					score += 1.0
			results[participant['condition']].append(score / len(participant['test_sequence']))
	return results

def mean_prop_correct_comprehension(dataset):
	results = { 'angle':[], 'size':[], 'both':[] }
	for participant in dataset:
		if participant['status'] == 'finished':
			score = 0.0
			for answer, response in zip(participant['test_sequence'], participant['test_responses']):
				if answer == partitions[participant['condition']][response]:
					score += 1.0
			results[participant['condition']].append(score / len(participant['test_sequence']))
	return results

def visualize_participants_production(dataset, dir_path):
	for participant in dataset:
		if participant['status'] == 'finished':
			partition = np.zeros(64, dtype=int)
			for answer, response in zip(participant['test_sequence'], participant['test_responses']):
				partition[answer] = response
			partition = partition.reshape((8,8))
			file_path = dir_path + '%s/%s.pdf' % (participant['condition'], participant['user_id'])
			visualize.visualize(partition, file_path)

prod_data = tools.read_json_lines('../data/experiments/exp1_production.json')
comp_data = tools.read_json_lines('../data/experiments/exp1_comprehension.json')

# prod_results = mean_prop_correct_production(prod_data)
# comp_results = mean_prop_correct_comprehension(comp_data)
# plot_prop_correct(prod_results, comp_results, '../visuals/exp1_results.svg')

# visualize_participants_production(prod_data, '../visuals/production/')

