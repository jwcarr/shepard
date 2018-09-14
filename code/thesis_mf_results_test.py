'''
Code for visualizing the model fit on simulated test data. This was
not included in the paper; refer to my thesis instead.
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from skopt import expected_minimum
import tools
from skopt import plots
import il_results
import colors

plt.rcParams['svg.fonttype'] = 'none' # don't convert fonts to curves

figure_layout = [['expressivity', 'error'],
                 ['complexity',   'cost' ]]

measure_bounds = {'expressivity':(1, 4), 'complexity':(0, 100), 'cost':(2, 4), 'error':(0, 3)}
measures_names = {'expressivity':'Expressivity', 'complexity':'Complexity', 'cost':'Comm. cost', 'error':'Trans. error'}

def load_optimizer(opt_path):
	with open(opt_path, mode='rb') as file:
		opt = pickle.load(file)
	return opt

def make_figure(datasets, opt, figure_path, n_levels=32):
	fig = plt.figure(figsize=(5.1, 1.9))
	gs = gridspec.GridSpec(2, 4, width_ratios=[8, 8, 10, 1])
	exp_axis = plt.subplot(gs[0,0])
	err_axis = plt.subplot(gs[0,1])
	com_axis = plt.subplot(gs[1,0])
	cst_axis = plt.subplot(gs[1,1])
	fit_axis = plt.subplot(gs[0:2,2])
	leg_axis = plt.subplot(gs[0:2,3])

	(dataset, label, color, color_conf, linestyle) = datasets

	for measure, axis in [('expressivity', exp_axis), ('error', err_axis), ('complexity', com_axis), ('cost', cst_axis)]:
		mean, conf_pos, conf_neg, start_gen = dataset[measure]
		xvals = list(range(start_gen, start_gen+len(mean)))
		axis.plot(xvals, mean, c=color, label=label, linestyle=linestyle, dash_capstyle="round", linewidth=2)
		axis.fill_between(xvals, conf_neg, conf_pos, facecolor=color_conf, alpha=0.5)

		ylim = measure_bounds[measure]
		ypad = (ylim[1]-ylim[0])*0.05
		axis.set_ylim(ylim[0]-ypad, ylim[1]+ypad)
		axis.set_ylabel(measures_names[measure], fontsize=8)
		axis.set_yticklabels([])
		axis.set_yticks([])
		for tick in axis.xaxis.get_major_ticks():
			tick.label.set_fontsize(7)
		axis.set_xlim(0,10)
		if measure == 'expressivity' or measure == 'error':
			axis.set_xticks([])
		else:
			axis.set_xticks([0,2,4,6,8,10])
			axis.set_xlabel('Generation', fontsize=8)

	(w_star, e_star), neg_log_p_sim = expected_minimum(opt)
	print('Simplicity:', len(opt.func_vals), 'iterations)')
	print('w* =', w_star)
	print('e* =', e_star)
	print('log(P) =', -neg_log_p_sim)

	space = opt.space
	samples = np.asarray(opt.x_iters)
	rvs_transformed = space.transform(space.rvs(n_samples=250))

	xi, yi, zi = plots.partial_dependence(space, opt.models[-1], 1, 0, rvs_transformed, 250)

	zmin = zi.min()
	zmax = zi.max()
	levels = np.geomspace(zmin, zmax, n_levels+1)

	cs = fit_axis.contourf(xi, yi, zi, levels, cmap='viridis_r')

	fit_axis.plot([0, w_star, w_star], [e_star, e_star, 0], c='k', linestyle=':')
	fit_axis.scatter(w_star, e_star, c='k', s=100, lw=0, marker='*')

	fit_axis.set_xlim(0,2)
	fit_axis.set_ylim(0,1)

	fit_axis.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
	for tick in fit_axis.xaxis.get_major_ticks():
		tick.label.set_fontsize(7)

	fit_axis.set_yticklabels([])
	fit_axis.set_yticks([])

	fit_axis.set_xlabel('Weight (w)', fontsize=8)
	fit_axis.set_ylabel('Noise (ε)', fontsize=8)

	cb = plt.colorbar(cs, cax=leg_axis)
	cb.set_ticks([])
	leg_axis.yaxis.set_label_position("left")
	leg_axis.set_ylabel('log likelihood', fontsize=8)
	leg_axis.invert_yaxis()

	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5, rect=(0.01, 0, 1, 1))
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

def make_model_results_figure(figure_path):
	model_results_sim = il_results.load('../data/test_modelfit/model/results.json', start_gen=0, end_gen=10, method='lang')
	dataset = (model_results_sim, 'Simplicity', colors.blue, colors.light_blue, '-')
	opt = load_optimizer('../data/test_modelfit/modelfit/result')
	make_figure(dataset, opt, figure_path=figure_path, n_levels=32)


make_model_results_figure('/Users/jon/Desktop/simulated_model_fit.eps')
