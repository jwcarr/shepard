import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from skopt import expected_minimum
import tools

plt.rcParams['svg.fonttype'] = 'none' # don't convert fonts to curves

def load_optimizer(opt_path):
	with open(opt_path, mode='rb') as file:
		opt = pickle.load(file)
	return opt

def get_gaussian_process(model, space, bounds, n_points=50):
	W = np.linspace(*bounds[0], n_points)
	E = np.linspace(0, 1, n_points)
	P = model.predict([(w, e) for e in E for w in W]).reshape((n_points, n_points))
	return W, E, P

def make_plot(sim_opt, inf_opt, figure_path, n_levels=32, show_evaluations=False, show_maximum=True, log_scale=True):
	'''
	Make contour plots showing the model fit under the simplicity and
	informativeness priors.
	'''
	fig = plt.figure(figsize=(5.5, 2.4))
	gs = gridspec.GridSpec(1, 3, width_ratios=[10, 10, 1])
	sim_axis, inf_axis, leg_axis = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])

	sim_xyz = get_gaussian_process(sim_opt['models'][-1], sim_opt.space, sim_opt.space.bounds)
	inf_xyz = get_gaussian_process(inf_opt['models'][-1], inf_opt.space, inf_opt.space.bounds)

	zmin = min(sim_xyz[2].min(), inf_xyz[2].min())
	zmax = max(sim_xyz[2].max(), inf_xyz[2].max())

	if log_scale:
		levels = np.geomspace(zmin, zmax, n_levels+1)
		# ticks = [2**i for i in range(int(np.log2(zmin)), int(np.log2(zmax))+2)]
	else:
		levels = np.linspace(zmin, zmax, n_levels+1)
		# ticks = [int(i) for i in np.linspace(levels[0], levels[-1], 7)]

	for axis, opt, xyz in [(sim_axis, sim_opt, sim_xyz), (inf_axis, inf_opt, inf_xyz)]:
		if log_scale:
			cs = axis.contourf(*xyz, levels, cmap='viridis_r', norm=colors.LogNorm())
		else:
			cs = axis.contourf(*xyz, levels, cmap='viridis_r')
		if show_evaluations:
			samples = np.asarray(opt.x_iters)
			axis.scatter(samples[:, 0], samples[:, 1], c='k', s=8, lw=0.)
		if show_maximum:
			(w_star, e_star), _ = expected_minimum(opt)
			axis.plot([0, w_star, w_star], [e_star, e_star, 0], c='k', linestyle=':')
			axis.scatter(*opt.x, c='k', s=100, lw=0, marker='*')
		axis.set_xlim(*opt.space.bounds[0])
		axis.set_ylim(0,1)
		axis.set_xlabel('Weight (w)')
	sim_axis.set_title('Simplicity prior (πsim)', fontsize=10)
	inf_axis.set_title('Informativeness prior (πinf)', fontsize=10)
	sim_axis.set_ylabel('Noise (ε)')
	inf_axis.set_yticklabels([])

	cb = plt.colorbar(cs, cax=leg_axis)
	cb.set_label('log likelihood (Eq. 12)', labelpad=-20, y=0.5)
	cb.set_ticks([levels[0], levels[-1]])
	cb.set_ticklabels(['high', 'low'])
	leg_axis.invert_yaxis()

	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5)
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

######################################################################

# sim_opt = load_optimizer('../data/modelfit/simplicity/result')
# (w_star, e_star), neg_log_p_sim = expected_minimum(sim_opt)
# print('Simplicity:', len(sim_opt.func_vals), 'iterations)')
# print('w* =', w_star)
# print('e* =', e_star)
# print('log(P) =', -neg_log_p_sim)

# inf_opt = load_optimizer('../data/modelfit/informativeness/result')
# (w_star, e_star), neg_log_p_inf = expected_minimum(inf_opt)
# print('\nInformativeness:', len(inf_opt.func_vals), 'iterations)')
# print('w* =', w_star)
# print('e* =', e_star)
# print('log(P) =', -neg_log_p_inf)

# print('\nLLR =', neg_log_p_inf - neg_log_p_sim)

# make_plot(sim_opt, inf_opt, figure_path='../manuscript/figs/exp2_model_fit.eps')
# make_plot(sim_opt, inf_opt, figure_path='../visuals/model_fit.pdf')
