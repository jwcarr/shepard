import matplotlib.pyplot as plt
from matplotlib import gridspec
import commcost
import colors
import tools

plt.rcParams['svg.fonttype'] = 'none' # don't convert fonts to curves in SVGs

def make_gaussians_figure(figure_path):

	gammas = [0.001, 0.1, 10]
	metric = 'circle_euclidean'
	y_max = 0.07

	N = commcost.Need((64,))
	P = commcost.Partition([0]*16 + [1]*16 + [2]*16 + [3]*16)
	S = P.spawn_speaker()

	fig = plt.figure(figsize=(5.5, 2))
	grid = gridspec.GridSpec(nrows=1, ncols=len(gammas))
	for i in range(len(gammas)):
		sp = fig.add_subplot(grid[0, i])
		sp.set_axisbelow(True)

		if gammas[i] == 'inf':
			L = P.spawn_listener(999999999)
			K = commcost.cost(P, N, S, L)
			plt.title('γ = ∞', fontsize=10)
		else:
			L = P.spawn_listener(gammas[i], metric)
			K = commcost.cost(P, N, S, L)
			plt.title('γ = ' + str(gammas[i]), fontsize=10)
		for j in range(len(P)):
			category_distribution = list(L[j].probabilities.flatten())
			category_distribution = [category_distribution[-1]] + category_distribution + [category_distribution[0]]
			sp.plot(range(0, P.size+2), category_distribution, linewidth=2.0, color=colors.categories[j])
		plt.xticks(fontsize=7)
		plt.yticks(fontsize=7)
		if i == 0:
			plt.ylabel('Probability', fontsize=10)
		else:
			sp.set_yticklabels([])
		if i == len(gammas)//2:
			plt.xlabel('Universe', fontsize=10)
		plt.xlim(1, P.size)
		y_max = y_max * 1.05
		y_min = 0 - (y_max * 0.05)
		plt.ylim(-0.005, 0.065)
		ticks = [1] + [(P.size//len(P))*j for j in range(1, len(P)+1)]
		sp.set_xticks(ticks)
	grid.tight_layout(fig, pad=0.1, h_pad=0.1, w_pad=0.5)
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)


make_gaussians_figure('../supplementary/S1_cost/gaussians.eps')
