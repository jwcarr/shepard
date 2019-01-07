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

rectlang_space = rectlang.Space((8,8))
commcost_space = commcost.Space((8,8))
commcost_space_discrete = commcost.Space((8,8), gamma=99999)

def save_results(array, filename):
	with open(filename, mode='wb') as file:
		pickle.dump(array, file)

def restore_results(filename):
	with open(filename, mode='rb') as file:
		array = pickle.load(file)
	return array

def sim_all_cats(n_sims, shape, convex=False):
	complexy_results = np.zeros((64, n_sims), dtype=float)
	commcost_results = np.zeros((64, n_sims), dtype=float)
	commcost_results_discrete = np.zeros((64, n_sims), dtype=float)
	for cat_i in range(64):
		for sim_i in range(n_sims):
			_, language = commcost.random_partition((8,8), cat_i+1, convex)
			complexy_results[cat_i,sim_i] = rectlang_space.complexity(language)
			commcost_results[cat_i,sim_i] = commcost_space.cost(language)
			commcost_results_discrete[cat_i,sim_i] = commcost_space_discrete.cost(language)
	return complexy_results, commcost_results, commcost_results_discrete

def run_sims(n_sims):
	non_complexy, non_commcost, non_commcost_discrete = sim_all_cats(n_sims, (8,8), convex=False)
	con_complexy, con_commcost, con_commcost_discrete = sim_all_cats(n_sims, (8,8), convex=True)
	save_results(non_complexy, '../data/sim_comp_cost/non_complexy')
	save_results(non_commcost, '../data/sim_comp_cost/non_commcost')
	save_results(non_commcost_discrete, '../data/sim_comp_cost/non_commcost_discrete')
	save_results(con_complexy, '../data/sim_comp_cost/con_complexy')
	save_results(con_commcost, '../data/sim_comp_cost/con_commcost')
	save_results(con_commcost_discrete, '../data/sim_comp_cost/con_commcost_discrete')

def plot_sims(figure_path, figsize=(5, 2.5)):
	non_commcost_results = restore_results('../data/sim_comp_cost/non_commcost')
	non_complexy_results = restore_results('../data/sim_comp_cost/non_complexy')
	con_commcost_results = restore_results('../data/sim_comp_cost/con_commcost')
	con_complexy_results = restore_results('../data/sim_comp_cost/con_complexy')

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
	non_commcost_disc = restore_results('../data/sim_comp_cost/non_commcost_discrete')
	con_commcost_disc = restore_results('../data/sim_comp_cost/con_commcost_discrete')
	non_commcost_cont = restore_results('../data/sim_comp_cost/non_commcost')
	con_commcost_cont = restore_results('../data/sim_comp_cost/con_commcost')

	fig, axes = plt.subplots(1, 2, figsize=figsize)
	axes[0].plot(range(1,65), non_commcost_disc.mean(axis=1), color=colors.categories[0], label='Random')
	axes[0].plot(range(1,65), con_commcost_disc.mean(axis=1), color=colors.categories[2], label='Convex')
	axes[1].plot(range(1,65), non_commcost_cont.mean(axis=1), color=colors.categories[0])
	axes[1].plot(range(1,65), con_commcost_cont.mean(axis=1), color=colors.categories[2])

	axes[1].annotate ('', (16, non_commcost_cont[15].mean()), (16, con_commcost_cont[15].mean()), arrowprops={'arrowstyle':'<->', 'shrinkA':0, 'shrinkB':0})
	
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
	non_complexy_results = restore_results('../data/sim_comp_cost/non_complexy')
	con_complexy_results = restore_results('../data/sim_comp_cost/con_complexy')

	fig, axis = plt.subplots(1, 1, figsize=figsize)
	axis.plot(range(1,65), non_complexy_results.mean(axis=1), color=colors.categories[0], label='Random')
	axis.plot(range(1,65), con_complexy_results.mean(axis=1), color=colors.categories[2], label='Convex')

	axis.annotate ('', (16, non_complexy_results[15].mean()), (16, con_complexy_results[15].mean()), arrowprops={'arrowstyle':'<->', 'shrinkA':0, 'shrinkB':0})
	
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

def reduce_points(X, Y):
	'''
	reduce the number of points in a scatter to reduce figure filesize
	'''
	X = [round(x, 2) for x in X]
	Y = [round(y, 2) for y in Y]
	unique = set(list(zip(X, Y)))
	return zip(*unique)

def plot_all(figure_path):
	cost_rand = restore_results('../data/sim_comp_cost/non_commcost')
	comp_rand = restore_results('../data/sim_comp_cost/non_complexy')
	cost_conv = restore_results('../data/sim_comp_cost/con_commcost')
	comp_conv = restore_results('../data/sim_comp_cost/con_complexy')

	fig, axis = plt.subplots(1, 1, figsize=(2.5,2.5))

	axis.scatter([comp_rand[0,0]], [cost_rand[0,0]], c='#E6CFDA', s=2.0)

	for i in range(1, 63):
		x, y = reduce_points(comp_rand[i], cost_rand[i])
		axis.scatter(x, y, c=colors.fake_alpha(colors.categories[0], (i+32)/95.), s=2.0)
	for i in range(1, 63):
		x, y = reduce_points(comp_conv[i], cost_conv[i])
		axis.scatter(x, y, c=colors.fake_alpha(colors.categories[2], (i+32)/95.), s=2.0)

	axis.scatter(comp_rand[63,0], cost_rand[63,0], c=colors.categories[0], s=2.0, label='Random')
	axis.scatter(comp_conv[63,0], cost_conv[63,0], c=colors.categories[2], s=2.0, label='Convex')

	axis.set_xlabel('Complexity (bits)', fontsize=8)
	axis.set_ylabel('Communicative cost (bits)', fontsize=8)
	for tick in axis.yaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axis.xaxis.get_major_ticks():
		tick.label.set_fontsize(7)

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=2, markerscale=3, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=8)

	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5, rect=[0,0.08,1,1])
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

def plot_advantage(figure_path, figsize=(5, 2.5)):
	non_commcost = restore_results('../data/sim_comp_cost/non_commcost')
	con_commcost = restore_results('../data/sim_comp_cost/con_commcost')
	non_complexy = restore_results('../data/sim_comp_cost/non_complexy')
	con_complexy = restore_results('../data/sim_comp_cost/con_complexy')

	complexy_adv = non_complexy.mean(1) - con_complexy.mean(1)
	commcost_adv = non_commcost.mean(1) - con_commcost.mean(1)

	fig, axes = plt.subplots(1, 2, figsize=figsize)
	axes[0].plot(range(1,65), complexy_adv, color='k')
	axes[1].plot(range(1,65), commcost_adv, color='k')
	
	axes[0].set_ylim(0-(400*0.025), 400+(400*0.025))
	axes[1].set_ylim(0-(0.7*0.025), 0.7+(0.7*0.025))

	axes[0].set_xlim(0,65)
	axes[1].set_xlim(0,65)
	axes[0].set_xticks([1, 16, 32, 48, 64])
	axes[1].set_xticks([1, 16, 32, 48, 64])
	axes[0].set_xlabel('Number of categories', fontsize=8)
	axes[1].set_xlabel('Number of categories', fontsize=8)
	# axes[0].set_yticks([0, 1, 2, 3, 4, 5, 6])
	axes[1].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
	axes[0].set_ylabel('Compactness advantage (bits)', fontsize=8)
	axes[0].set_title('Complexity', fontsize=8)
	axes[1].set_title('Communicative cost', fontsize=8)
	for tick in axes[0].yaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axes[0].xaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axes[1].yaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	for tick in axes[1].xaxis.get_major_ticks():
		tick.label.set_fontsize(7)
	
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5)
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

# run_sims(100)

# plot_sims('../visuals/simulated_complexity_cost.svg')

# plot_cost_sims('../visuals/simulated_cost.svg')

# plot_comp_sims('../visuals/simulated_comp.svg')

# plot_all('../visuals/simulated_complexity_cost.svg')

# plot_advantage('../visuals/compactness_advantage.svg')