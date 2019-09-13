import json
import numpy as np
import matplotlib.pyplot as plt
import tools

plt.rcParams['svg.fonttype'] = 'none' # don't convert fonts to curves in SVGs
plt.rcParams.update({'font.size': 7})

figure_layout = [['expressivity','complexity', 'cost', 'error']]

measure_bounds = {'expressivity':(1, 4), 'complexity':(0, 600), 'cost':(4, 6), 'error':(0, 4)}
measures_names = {'expressivity':'Number of categories', 'complexity':'Complexity (bits)', 'cost':'Communicative cost (bits)', 'error':'Transmission error (bits)'}

def distance(x, y):
	return np.sqrt(sum((x-y)**2 for x, y in zip(x, y)))

def load(data_path, start_gen, end_gen, method='prod', return_typical_chain=False):
	data = tools.read_json_file(data_path)
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

def extract_generation_distribution(data_path, measure, generation):
	data = tools.read_json_file(data_path)
	distribution = []
	for chain in data['chains']:
		datum = chain['generations'][generation][measure]
		distribution.append(datum)
	return distribution

def make_figure(datasets, figure_path, title=None, show_legend=False, deep_legend=False, figsize=None):
	if show_legend:
		if deep_legend:
			if figsize is None:
				figsize = (5.5, 5.0)
			fig, axes = plt.subplots(len(figure_layout), len(figure_layout[0]), figsize=figsize, squeeze=False, sharex=True)
		else:
			if figsize is None:
				figsize = (7.2, 1.75)
			fig, axes = plt.subplots(len(figure_layout), len(figure_layout[0]), figsize=figsize, squeeze=False)
	else:
		if figsize is None:
			figsize = (7.2, 1.6)
		fig, axes = plt.subplots(len(figure_layout), len(figure_layout[0]), figsize=figsize, squeeze=False)
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
		x_max = len(dataset[figure_layout[i][j]][0])-1
		if figure_layout[i][j] == 'error':
			x_max += 1
		axis.set_xlim(0, x_max)
		axis.set_xticks(list(range(0, x_max+1, x_max//5)))
		if i == len(axes) - 1:
			axis.set_xlabel('Generation')
	if title:
		fig.suptitle(title)
	if show_legend:
		if deep_legend:
			fig.legend(handles, labels, loc='lower center', ncol=1, frameon=False)
			fig.tight_layout(pad=2, h_pad=0.5, w_pad=0.5, rect=(0, 0.1, 1, 1))
		else:
			legend_offset = 1 / (figsize[1] / 0.15)
			if len(figure_layout) == 1:
				fig.legend(handles, labels, ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), loc='lower center')
			else:
				fig.legend(handles, labels, ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), loc='lower center')
			fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5, rect=(0.01, legend_offset, 1, 1))
	else:
		fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5, rect=(0.01, 0, 1, 1))
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

def make_carstensen_figure(datasets, figure_path, figsize):
	fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
	y_bounds = (4.6, 5)
	y_pad = (y_bounds[1]-y_bounds[0])*0.05
	
	for k, (dataset, label, color, color_conf, linestyle) in enumerate(datasets):
		mean, conf_pos, conf_neg, start_gen = dataset['cost']
		xvals = list(range(start_gen, start_gen+len(mean)))
		axes[0].plot(xvals, mean, c=color, label=label, linestyle=linestyle, dash_capstyle='round', linewidth=2)
		axes[0].fill_between(xvals, conf_neg, conf_pos, facecolor=color_conf, alpha=0.5)
	handles, labels = axes[0].get_legend_handles_labels()
	axes[0].set_ylim(y_bounds[0]-y_pad, y_bounds[1]+y_pad)
	axes[0].set_ylabel('Communicative cost (bits)')
	axes[0].set_xlim(0, 10)
	axes[0].set_xticks(list(range(0, 11, 2)))
	axes[0].set_xlabel('Generation')

	for k, (dataset, label, color, color_conf, linestyle) in enumerate(datasets):
		mean, conf_pos, conf_neg, start_gen = dataset['cost']
		xvals = list(range(start_gen, start_gen+len(mean)))
		axes[1].plot(xvals, mean, c=color, label=label, linestyle=linestyle, dash_capstyle='round', linewidth=2)
		axes[1].fill_between(xvals, conf_neg, conf_pos, facecolor=color_conf, alpha=0.5)
	axes[1].set_ylim(y_bounds[0]-y_pad, y_bounds[1]+y_pad)
	axes[1].set_xlim(0, 100)
	axes[1].set_xticks(list(range(0, 101, 20)))
	axes[1].set_xlabel('Generation')

	legend_offset = 1 / (figsize[1] / 0.15)
	fig.legend(handles, labels, ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), loc='lower center')
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5, rect=(0.01, legend_offset, 1, 1))
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

def plot_final_gen_densities(axis, results):
	positions = [0.6, 0.3, 0]
	y_lim = [0, 0.9]
	labels, colors, distributions = zip(*results)
	violins = axis.violinplot(distributions, positions, vert=False, showmedians=False, showextrema=False)
	for i, body in enumerate(violins['bodies']):
		m = np.mean(body.get_paths()[0].vertices[:, 1])
		body.get_paths()[0].vertices[:, 1] = np.clip(body.get_paths()[0].vertices[:, 1], m, np.inf)
		body.set_facecolor(colors[i])
		body.set_edgecolor(colors[i])
		body.set_alpha(1.0)
		axis.text(np.median(distributions[i]), positions[i]+0.1, labels[i], {'color':'white'}, ha='center', va='top')
	axis.set_yticklabels([])
	axis.tick_params(axis='y', which='both', left='off', right='off')
	axis.set_ylim(*y_lim)

def plot_final_gen_distributions(datasets, figure_path):
	fig, axes = plt.subplots(1, len(datasets), figsize=(7.2, 1.75))
	for i, (label, dataset) in enumerate(datasets):
		plot_final_gen_densities(axes[i], dataset)
		axes[i].set_title(label, fontsize=7)
	axes[0].set_xticks([25, 75, 125])
	axes[1].set_xticks([25, 75, 125])
	axes[2].set_xticks([25, 75, 125])
	axes[0].set_xlim(0, 150)
	axes[1].set_xlim(0, 150)
	axes[2].set_xlim(0, 150)
	axes[1].set_xlabel('Complexity')
	axes[3].set_xticks([4.25, 4.3, 4.35])
	axes[4].set_xticks([4.25, 4.3, 4.35])
	axes[5].set_xticks([4.25, 4.3, 4.35])
	axes[3].set_xlim(4.2, 4.4)
	axes[4].set_xlim(4.2, 4.4)
	axes[5].set_xlim(4.2, 4.4)
	axes[4].set_xlabel('Cost')
	fig.tight_layout(pad=0.01, h_pad=0.01, w_pad=0.01)
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)
