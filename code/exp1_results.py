import numpy as np
import matplotlib.pyplot as plt
import visualize
import colors
import tools

plt.rcParams['svg.fonttype'] = 'none'

partitions = {
	'angle': {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1,26:1,27:1,28:1,29:1,30:1,31:1,32:2,33:2,34:2,35:2,36:2,37:2,38:2,39:2,40:2,41:2,42:2,43:2,44:2,45:2,46:2,47:2,48:3,49:3,50:3,51:3,52:3,53:3,54:3,55:3,56:3,57:3,58:3,59:3,60:3,61:3,62:3,63:3},
	'size' : {0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3,8:0,9:0,10:1,11:1,12:2,13:2,14:3,15:3,16:0,17:0,18:1,19:1,20:2,21:2,22:3,23:3,24:0,25:0,26:1,27:1,28:2,29:2,30:3,31:3,32:0,33:0,34:1,35:1,36:2,37:2,38:3,39:3,40:0,41:0,42:1,43:1,44:2,45:2,46:3,47:3,48:0,49:0,50:1,51:1,52:2,53:2,54:3,55:3,56:0,57:0,58:1,59:1,60:2,61:2,62:3,63:3},
	'both' : {0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:1,8:0,9:0,10:0,11:0,12:1,13:1,14:1,15:1,16:0,17:0,18:0,19:0,20:1,21:1,22:1,23:1,24:0,25:0,26:0,27:0,28:1,29:1,30:1,31:1,32:2,33:2,34:2,35:2,36:3,37:3,38:3,39:3,40:2,41:2,42:2,43:2,44:3,45:3,46:3,47:3,48:2,49:2,50:2,51:2,52:3,53:3,54:3,55:3,56:2,57:2,58:2,59:2,60:3,61:3,62:3,63:3}
}

condition_names = {'angle':'Angle-only', 'both':'Angle &amp; Size', 'size':'Size-only'}

def generate_csv_for_stats(dataset, output_file):
	csv = 'subject,test_type,category_system,correct\n'
	subject = 1
	for participant in dataset:
		if participant['status'] == 'finished':
			if participant['test_type'] == 'production':
				for answer, response in zip(participant['test_sequence'], participant['test_responses']):
					if partitions[participant['condition']][answer] == response:
						csv += '%i,%s,%s,1\n' % (subject, participant['test_type'], participant['condition']) # correct
					else:
						csv += '%i,%s,%s,0\n' % (subject, participant['test_type'], participant['condition']) # incorrect
			elif participant['test_type'] == 'comprehension':
				for answer, response in zip(participant['test_sequence'], participant['test_responses']):
					if answer == partitions[participant['condition']][response]:
						csv += '%i,%s,%s,1\n' % (subject, participant['test_type'], participant['condition']) # correct
					else:
						csv += '%i,%s,%s,0\n' % (subject, participant['test_type'], participant['condition']) # incorrect
			else:
				raise ValueError('Unrecognized test type')
			subject += 1
	with open(output_file, mode='w') as file:
		file.write(csv)

def plot_densities(axis, results):
	positions = [0.6, 0.3, 0]
	y_lim = [0, 0.9]
	distributions = [results['angle'], results['size'], results['both']]
	axis.plot([0.25, 0.25], y_lim, c='black', linestyle='--', zorder=0)
	violins = axis.violinplot(distributions, positions, vert=False, showmedians=True, showextrema=False)
	for body in violins['bodies']:
		m = np.mean(body.get_paths()[0].vertices[:, 1])
		body.get_paths()[0].vertices[:, 1] = np.clip(body.get_paths()[0].vertices[:, 1], m, np.inf)
		body.set_facecolor(colors.black)
		body.set_edgecolor(colors.black)
		body.set_alpha(1.0)
	violins['cmedians'].set_color('white')
	axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
	axis.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
	axis.set_yticklabels([])
	axis.tick_params(axis='y', which='both', left='off', right='off')
	axis.set_xlabel('Proportion correct')
	axis.set_xlim(0,1)
	axis.set_ylim(*y_lim)

def plot_prop_correct(production_results, comprehension_results, figure_path):
	fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
	plot_densities(axes[0], production_results)
	plot_densities(axes[1], comprehension_results)
	axes[0].set_title('Production', fontsize=10)
	axes[1].set_title('Comprehension', fontsize=10)
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5)
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

def plot_click_entropy_all_experiments(figure_path):
	exp1 = tools.read_json_lines('../data/experiments/exp1_participants.json')
	exp2 = tools.read_json_lines('../data/experiments/exp2_participants.json')
	res1p = click_entropy(exp1, 'production')
	res1c = click_entropy(exp1, 'comprehension')
	res2 = click_entropy(exp2)
	fig, axes = plt.subplots(1, 3, figsize=(5.5, 2))

	y, c = zip(*res1p)
	axes[0].scatter(range(len(res1p)), y, c=c, marker='x')
	axes[0].set_ylim(-0.1,2.1); axes[0].set_xlim(0,len(res1p))
	axes[0].set_xticks([]); axes[0].set_xticklabels([])
	axes[0].set_title('Experiment 1 (P)', fontsize=10)
	
	y, c = zip(*res1c)
	axes[1].scatter(range(len(res1c)), y, c=c, marker='x')
	axes[1].set_ylim(0,6); axes[1].set_xlim(0,len(res1p))
	axes[1].set_xticks([]); axes[1].set_xticklabels([])
	axes[1].set_title('Experiment 1 (C)', fontsize=10)
	
	y, c = zip(*res2)
	axes[2].scatter(range(len(res2)), y, c=c, marker='x')
	axes[2].plot([0, len(res1p)], [1.75,1.75], c='red', linestyle='--')
	axes[2].set_ylim(-0.1,2.1); axes[2].set_xlim(0,len(res1p))
	axes[2].set_xticks([]); axes[2].set_xticklabels([])
	axes[2].set_title('Experiment 2', fontsize=10)

	axes[0].set_ylabel('Button-click entropy')
	axes[1].set_xlabel('Participant')
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5)
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

def click_entropy(dataset, test_type=None):
	results = []
	if test_type == 'production':
		for participant in dataset:
			if participant['test_type'] == 'production':
				H = entropy(participant['test_positions'], 4)
				if participant['status'] == 'finished':
					results.append((H, 'black'))
				else:
					results.append((H, 'red'))
	elif test_type == 'comprehension':
		for participant in dataset:
			if participant['test_type'] == 'comprehension':
				H = entropy(participant['test_positions'], 64)
				if participant['status'] == 'finished':
					results.append((H, 'black'))
				else:
					results.append((H, 'red'))
	else:
		for participant in dataset:
			H = entropy(participant['test_positions'], 4)
			if participant['status'] == 'finished':
				results.append((H, 'black'))
			else:
				results.append((H, 'red'))
	return results

def entropy(positions, n_buttons):
	n_clicks = len(positions)
	buttons = [positions.count(button_i) for button_i in range(n_buttons)]
	summation = 0.0
	for button in buttons:
		p = button / n_clicks
		if p > 0.0:
			summation += p * np.log2(p)
	return -summation

def mean_prop_correct_production(dataset):
	results = { 'angle':[], 'size':[], 'both':[] }
	for participant in dataset:
		if participant['test_type'] == 'production' and participant['status'] == 'finished':
			score = 0.0
			for answer, response in zip(participant['test_sequence'], participant['test_responses']):
				if partitions[participant['condition']][answer] == response:
					score += 1.0
			results[participant['condition']].append(score / len(participant['test_sequence']))
	return results

def mean_prop_correct_comprehension(dataset):
	results = { 'angle':[], 'size':[], 'both':[] }
	for participant in dataset:
		if participant['test_type'] == 'comprehension' and participant['status'] == 'finished':
			score = 0.0
			for answer, response in zip(participant['test_sequence'], participant['test_responses']):
				if answer == partitions[participant['condition']][response]:
					score += 1.0
			results[participant['condition']].append(score / len(participant['test_sequence']))
	return results

def visualize_participant_production(dataset, dir_path):
	for participant in dataset:
		if participant['test_type'] == 'production' and participant['status'] == 'finished':
			partition = np.zeros(64, dtype=int)
			for answer, response in zip(participant['test_sequence'], participant['test_responses']):
				partition[answer] = response
			partition = partition.reshape((8,8))
			figure_path = dir_path + '%s/%s.pdf' % (participant['condition'], participant['user_id'])
			visualize.visualize(partition, figure_path)

def visualize_participant_comprehension(dataset, dir_path):
	for participant in dataset:
		if participant['test_type'] == 'comprehension' and participant['status'] == 'finished':
			partition = np.zeros((4,8,8), dtype=float)
			for answer, response in zip(participant['test_sequence'], participant['test_responses']):
				row = response % 8
				col = response // 8
				partition[answer,row,col] += 1
			figure_path = dir_path + '%s/%s.pdf' % (participant['condition'], participant['user_id'])
			for cat in range(4):
				partition[cat] = partition[cat] / partition[cat].max()
			visualize.visualize(partition, figure_path)

def visualize_all_participant_production(dataset, figure_dir):
	partitions_by_condition = {'angle':[], 'both':[], 'size':[]}
	for participant in dataset:
		if participant['test_type'] == 'production' and participant['status'] == 'finished':
			partition = np.zeros(64, dtype=int)
			for answer, response in zip(participant['test_sequence'], participant['test_responses']):
				partition[answer] = response
			partition = partition.reshape((8,8))
			partitions_by_condition[participant['condition']].append(partition)
	for condition, partitions in partitions_by_condition.items():
		label = 'Production, ' + condition_names[condition]
		visualize.visualize_all(partitions, figure_dir + condition + '.pdf', test_type='production', label=label)

def visualize_all_participant_comprehension(dataset, figure_dir):
	partitions_by_condition = {'angle':[], 'both':[], 'size':[]}
	for participant in dataset:
		if participant['test_type'] == 'comprehension' and participant['status'] == 'finished':
			partition = np.zeros((4,8,8), dtype=float)
			for answer, response in zip(participant['test_sequence'], participant['test_responses']):
				row = response % 8
				col = response // 8
				partition[answer,row,col] += 1
			for cat in range(4):
				partition[cat] = partition[cat] / partition[cat].max()
			partitions_by_condition[participant['condition']].append(partition)
	for condition, partitions in partitions_by_condition.items():
		label = 'Comprehension, ' + condition_names[condition]
		visualize.visualize_all(partitions, figure_dir + condition + '.pdf', test_type='comprehension', label=label)

######################################################################

# dataset = tools.read_json_lines('../data/experiments/exp1_participants.json')

# generate_csv_for_stats(dataset, '../data/experiments/exp1_stats.csv')

# prod_results = mean_prop_correct_production(dataset)
# comp_results = mean_prop_correct_comprehension(dataset)
# plot_prop_correct(prod_results, comp_results, '../visuals/exp1_results.svg')

# visualize_participant_production(dataset, '../visuals/production/')
# visualize_participant_comprehension(dataset, '../visuals/comprehension/')

# visualize_all_participant_production(dataset, '../visuals/production_')
# visualize_all_participant_comprehension(dataset, '../visuals/comprehension_')

# plot_click_entropy_all_experiments('../supplementary/S3_participant_nums/exclusion.eps')
