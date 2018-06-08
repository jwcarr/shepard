import json
import numpy as np
import matplotlib.pyplot as plt
import tools

plt.rcParams['svg.fonttype'] = 'none' # don't convert fonts to curves in SVGs

figure_layout = [['expressivity', 'error'],
                 ['complexity',   'cost' ]]

measure_bounds = {'expressivity':(1, 4), 'complexity':(0, 600), 'cost':(4, 6), 'error':(0, 4)}
measures_names = {'expressivity':'Expressivity', 'complexity':'Complexity', 'cost':'Communicative cost', 'error':'Transmission error'}

def distance(x, y):
	return np.sqrt(sum((x-y)**2 for x, y in zip(x, y)))

def load(json_file, start_gen, end_gen, method='prod', return_typical_chain=False):
	data = tools.read_json_file(json_file)
	dataset = {}
	for measure in ['expressivity', 'complexity', 'cost', 'error']:
		dataset[measure] = extract_dataset(data, start_gen, end_gen, method + '_' + measure, return_typical_chain)
	return dataset

def extract_dataset(data, start_gen, end_gen, measure, return_typical_chain=False):
	if measure.endswith('error') and start_gen == 0:
		start_gen += 1
	n_gens = end_gen - start_gen + 1
	results = np.zeros((len(data['chains']), n_gens), dtype=float)
	for chain_i, chain in enumerate(data['chains']):
		for gen_i in range(start_gen, end_gen+1):
			try:
				results[chain_i, gen_i-start_gen] = chain['generations'][gen_i][measure]
			except IndexError:
				results[chain_i, gen_i-start_gen] = chain['generations'][chain['first_fixation']][measure]
	mean, std = results.mean(axis=0), results.std(axis=0)
	conf_interval = 1.96 * (std / np.sqrt(len(results)))
	conf_pos, conf_neg = mean + conf_interval, mean - conf_interval
	if not return_typical_chain:
		return mean, conf_pos, conf_neg, start_gen
	best_dist, best_chain = 9999999999.9, None
	for chain_i, chain in enumerate(results):
		dist = distance(mean, chain)
		if dist < best_dist:
			best_dist, best_chain = dist, chain_i
	return data['chains'][best_chain]

def extract_generation_distribution(file_path, measure, generation):
	data = tools.read_json_file(file_path)
	distribution = []
	for chain in data['chains']:
		datum = chain['generations'][generation][measure]
		distribution.append(datum)
	return distribution

def make_figure(datasets, file_path, title=None, show_legend=False, deep_legend=False):
	if show_legend:
		if deep_legend:
			fig, axes = plt.subplots(2, 2, figsize=(5.5, 5))
		else:
			fig, axes = plt.subplots(2, 2, figsize=(5.5, 4))
	else:
		fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.6))
	for (i, j), axis in np.ndenumerate(axes):
		measure = figure_layout[i][j]
		for k, (dataset, label, color, color_conf, linestyle) in enumerate(datasets):
			mean, conf_pos, conf_neg, start_gen = dataset[measure]
			xvals = list(range(start_gen, start_gen+len(mean)))
			axis.plot(xvals, mean, c=color, label=label, linestyle=linestyle, dash_capstyle="round", linewidth=2)
			axis.fill_between(xvals, conf_neg, conf_pos, facecolor=color_conf, alpha=0.5)
		handles, labels = axis.get_legend_handles_labels()
		ylim = measure_bounds[measure]
		ypad = (ylim[1]-ylim[0])*0.05
		axis.set_ylim(ylim[0]-ypad, ylim[1]+ypad)
		axis.set_ylabel(measures_names[measure])
		axis.set_xlim(0, xvals[-1])
		axis.set_xticks(list(range(0, xvals[-1]+1, xvals[-1]//5)))
		if i == 0:
			axis.set_xticklabels([])
		else:
			axis.set_xlabel('Generation')
	# Line up the y-axis labels
	for axis in axes[:, 0]:
		d = axis.get_yaxis().set_label_coords(-0.2, 0.5)
	for axis in axes[:, 1]:
		d = axis.get_yaxis().set_label_coords(-0.2, 0.5)
	if title:
		fig.suptitle(title)
	if show_legend:
		if deep_legend:
			fig.legend(handles, labels, loc='lower center', ncol=1, frameon=False)
			fig.tight_layout(pad=2, h_pad=0.5, w_pad=0.5, rect=(0, 0.1, 1, 1))
		else:
			fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
			fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5, rect=(0, 0.1, 1, 1))
	else:
		fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
	fig.savefig(file_path, format='svg')
	tools.format_svg_labels(file_path)
	if not file_path.endswith('.svg'):
		tools.convert_svg(file_path, file_path)

def plot_final_gen_densities(axis, results):
	positions = [0, -0.4, -0.8]
	distributions = list(results.values())
	labels = list(results.keys())
	violins = axis.violinplot(distributions, positions, vert=False, showmedians=False, showextrema=False)
	for i, body in enumerate(violins['bodies']):
		m = np.mean(body.get_paths()[0].vertices[:, 1])
		body.get_paths()[0].vertices[:, 1] = np.clip(body.get_paths()[0].vertices[:, 1], m, np.inf)
		body.set_facecolor('#323536')
		body.set_edgecolor('#323536')
		body.set_alpha(1.0)
		axis.text(np.median(distributions[i]), positions[i]+0.1, labels[i], {'color':'white'}, ha='center', va='top')
	axis.set_yticklabels([])
	axis.tick_params(axis='y', which='both', left='off', right='off')
	axis.set_xlabel('Complexity')
	max_x = max([max(distribution) for distribution in distributions])
	axis.set_xlim(0, max_x)
	axis.set_ylim(-0.8, 0.3)

def plot_final_gen_distributions(bottleneck_results, exposure_results, noise_results, file_path):
	fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.5))
	plot_final_gen_densities(axes[0], bottleneck_results)
	plot_final_gen_densities(axes[1], exposure_results)
	plot_final_gen_densities(axes[2], noise_results)
	axes[0].set_title('Bottleneck', fontsize=10)
	axes[1].set_title('Exposures', fontsize=10)
	axes[2].set_title('Noise', fontsize=10)
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5)
	fig.savefig(file_path, format='svg')
	tools.format_svg_labels(file_path)
	if not file_path.endswith('.svg'):
		tools.convert_svg(file_path, file_path)
