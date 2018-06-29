import numpy as np
import matplotlib.pyplot as plt
import colors
import tools

plt.rcParams['svg.fonttype'] = 'none'

def make_fig(figure_path):

	comp_W = np.linspace(0,4,100)

	comp_qudrnt = np.array([-39.25512476486815*i for i in comp_W])
	comp_stripe = np.array([-35.90911969399966*i for i in comp_W])
	comp_sum = np.logaddexp2(comp_stripe, comp_qudrnt)

	cost_W = np.linspace(0,100,100)

	cost_qudrnt = np.array([-4.200174400715539*i for i in cost_W])
	cost_stripe = np.array([-4.294113843380405*i for i in cost_W])
	cost_sum = np.logaddexp2(cost_stripe, cost_qudrnt)

	fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)

	axes[0].plot(comp_W, 2**(comp_stripe-comp_sum), c=colors.black, linestyle='-', label='Stripes')
	axes[0].plot(comp_W, 2**(comp_qudrnt-comp_sum), c=colors.black, linestyle='--', label='Quadrants')
	axes[0].set_ylabel('Prior probability')
	axes[0].set_xlabel('Weight (w)')
	axes[0].set_title('Simplicity prior (πsim)', fontsize=10)
	axes[0].set_xlim(0, comp_W[-1])

	axes[1].plot(cost_W, 2**(cost_stripe-cost_sum), c=colors.black, linestyle='-')
	axes[1].plot(cost_W, 2**(cost_qudrnt-cost_sum), c=colors.black, linestyle='--')
	axes[1].set_xlabel('Weight (w)')
	axes[1].set_title('Informativeness prior (πinf)', fontsize=10)
	axes[1].set_xlim(0, cost_W[-1])

	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
	fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5, rect=(0, 0.15, 1, 1))
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

make_fig('../manuscript/figs/exp1_prior_comparison.eps')
