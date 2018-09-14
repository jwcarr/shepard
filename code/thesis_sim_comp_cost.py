'''
Code for simulating the complexity and cost of randomly generated
languages. This was not included in the paper; refer to my thesis
instead.
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
import commcost
import rectlang
import tools
import colors

plt.rcParams['svg.fonttype'] = 'none'

commcost_space = commcost.Space((8,8))
rectlang_space = rectlang.Space((8,8))

def save_results(array, filename):
	with open(filename, mode='wb') as file:
		pickle.dump(array, file)

def restore_results(filename):
	with open(filename, mode='rb') as file:
		array = pickle.load(file)
	return array

def sim_all_cats(n_sims, shape, convex=False):
	commcost_results = np.zeros((64, n_sims), dtype=float)
	complexy_results = np.zeros((64, n_sims), dtype=float)
	for cat_i in range(64):
		for sim_i in range(n_sims):
			_, language = commcost.random_partition((8,8), cat_i+1, convex)
			commcost_results[cat_i,sim_i] = commcost_space.cost(language)
			complexy_results[cat_i,sim_i] = rectlang_space.complexity(language)
	return commcost_results, complexy_results

def run_2d_sims(n_sims):
	non_commcost_results, non_complexy_results = sim_all_cats(n_sims, (8,8), convex=False)
	con_commcost_results, con_complexy_results = sim_all_cats(n_sims, (8,8), convex=True)
	save_results(non_commcost_results, '/Users/jon/Desktop/non_commcost')
	save_results(non_complexy_results, '/Users/jon/Desktop/non_complexy')
	save_results(con_commcost_results, '/Users/jon/Desktop/con_commcost')
	save_results(con_complexy_results, '/Users/jon/Desktop/con_complexy')

def plot_sims(figure_path, figsize=(5, 2.5)):
	non_commcost_results = restore_results('/Users/jon/Desktop/non_commcost')
	non_complexy_results = restore_results('/Users/jon/Desktop/non_complexy')
	con_commcost_results = restore_results('/Users/jon/Desktop/con_commcost')
	con_complexy_results = restore_results('/Users/jon/Desktop/con_complexy')

	fig, axes = plt.subplots(1, 2, figsize=figsize)
	axes[0].plot(range(1,65), non_complexy_results.mean(axis=1), color=colors.categories[0], label='Random')
	axes[0].plot(range(1,65), con_complexy_results.mean(axis=1), color=colors.categories[2], label='Convex')
	axes[1].plot(range(1,65), non_commcost_results.mean(axis=1), color=colors.categories[0])
	axes[1].plot(range(1,65), con_commcost_results.mean(axis=1), color=colors.categories[2])
	
	axes[0].set_ylim(0-(700*0.025), 700+(700*0.025))
	axes[1].set_ylim(0-(6*0.025), 6+(6*0.025))

	axes[0].set_xlim(0,65)
	axes[1].set_xlim(0,65)
	axes[0].set_xticks([1, 16, 32, 48, 64])
	axes[1].set_xticks([1, 16, 32, 48, 64])
	axes[0].set_xlabel('Number of categories', fontsize=8)
	axes[1].set_xlabel('Number of categories', fontsize=8)
	axes[0].set_yticks([0, 100, 200, 300, 400, 500, 600, 700])
	axes[1].set_yticks([0, 1, 2, 3, 4, 5, 6])
	axes[0].set_ylabel('Complexity (bits)', fontsize=8)
	axes[1].set_ylabel('Communicative cost (bits)', fontsize=8)
	for tick in axes[0].yaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axes[0].xaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axes[1].yaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axes[1].xaxis.get_major_ticks():
		tick.label.set_fontsize(7)

	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=8)
	
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5, rect=[0,0.08,1,1])
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)


def plot_cost_sims(figure_path, figsize=(5, 2.5)):
	non_commcost_disc = restore_results('/Users/jon/Desktop/non_commcost_discrete')
	con_commcost_disc = restore_results('/Users/jon/Desktop/con_commcost_discrete')
	non_commcost_cont = restore_results('/Users/jon/Desktop/non_commcost')
	con_commcost_cont = restore_results('/Users/jon/Desktop/con_commcost')

	fig, axes = plt.subplots(1, 2, figsize=figsize)
	axes[0].plot(range(1,65), non_commcost_disc.mean(axis=1), color=colors.categories[0], label='Random')
	axes[0].plot(range(1,65), con_commcost_disc.mean(axis=1), color=colors.categories[2], label='Convex')
	axes[1].plot(range(1,65), non_commcost_cont.mean(axis=1), color=colors.categories[0])
	axes[1].plot(range(1,65), con_commcost_cont.mean(axis=1), color=colors.categories[2])
	
	axes[0].set_ylim(0-(6*0.025), 6+(6*0.025))
	axes[1].set_ylim(0-(6*0.025), 6+(6*0.025))

	axes[0].set_xlim(0,65)
	axes[1].set_xlim(0,65)
	axes[0].set_xticks([1, 16, 32, 48, 64])
	axes[1].set_xticks([1, 16, 32, 48, 64])
	axes[0].set_xlabel('Number of categories', fontsize=8)
	axes[1].set_xlabel('Number of categories', fontsize=8)
	axes[0].set_yticks([0, 1, 2, 3, 4, 5, 6])
	axes[1].set_yticks([0, 1, 2, 3, 4, 5, 6])
	axes[0].set_ylabel('Communicative cost (bits)', fontsize=8)
	for tick in axes[0].yaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axes[0].xaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axes[1].yaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axes[1].xaxis.get_major_ticks():
		tick.label.set_fontsize(7)

	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=8)
	
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5, rect=[0,0.08,1,1])
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)


def plot_comp_sims(figure_path, figsize=(2.5, 2.5)):
	non_complexy_results = restore_results('/Users/jon/Desktop/non_complexy')
	con_complexy_results = restore_results('/Users/jon/Desktop/con_complexy')

	fig, axis = plt.subplots(1, 1, figsize=figsize)
	axis.plot(range(1,65), non_complexy_results.mean(axis=1), color=colors.categories[0], label='Random')
	axis.plot(range(1,65), con_complexy_results.mean(axis=1), color=colors.categories[2], label='Convex')
	
	axis.set_ylim(0-(700*0.025), 700+(700*0.025))

	axis.set_xlim(0,65)
	axis.set_xticks([1, 16, 32, 48, 64])
	axis.set_xlabel('Number of categories', fontsize=8)
	axis.set_yticks([0, 100, 200, 300, 400, 500, 600, 700])
	axis.set_ylabel('Complexity (bits)', fontsize=8)
	for tick in axis.yaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axis.xaxis.get_major_ticks():
		tick.label.set_fontsize(7)

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=8)
	
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5, rect=[0,0.08,1,1])
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

def plot_all(figure_path):
	cost_rand = restore_results('/Users/jon/Desktop/non_commcost')
	comp_rand = restore_results('/Users/jon/Desktop/non_complexy')
	cost_conv = restore_results('/Users/jon/Desktop/con_commcost')
	comp_conv = restore_results('/Users/jon/Desktop/con_complexy')

	fig, axis = plt.subplots(1, 1, figsize=(2.5,2.5))

	for i in range(63):
		axis.scatter(comp_rand[i,:10], cost_rand[i,:10], c=colors.fake_alpha(colors.categories[0], (i+32)/95.), s=2.0)
	for i in range(63):
		axis.scatter(comp_conv[i,:10], cost_conv[i,:10], c=colors.fake_alpha(colors.categories[2], (i+32)/95.), s=2.0)

	axis.scatter(comp_rand[i,:10], cost_rand[i,:10], c=colors.categories[0], s=2.0, label='Random')
	axis.scatter(comp_conv[i,:10], cost_conv[i,:10], c=colors.categories[2], s=2.0, label='Convex')

	axis.set_xlabel('Complexity (bits)', fontsize=8)
	axis.set_ylabel('Communicative cost (bits)', fontsize=8)
	for tick in axis.yaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axis.xaxis.get_major_ticks():
		tick.label.set_fontsize(7)

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=8)

	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5, rect=[0,0.08,1,1])
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

# run_sims(100)

# plot_sims('/Users/jon/Desktop/simulated_complexity_cost.eps')

# plot_cost_sims('/Users/jon/Desktop/simulated_cost.eps')

# plot_comp_sims('/Users/jon/Desktop/simulated_comp.eps')

# plot_all('/Users/jon/Desktop/simulated_complexity_cost.eps')
