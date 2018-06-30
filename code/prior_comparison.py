import numpy as np
import matplotlib.pyplot as plt
import colors
import tools

plt.rcParams['svg.fonttype'] = 'none'

def make_fig(figure_path):

	comp_weights = np.linspace(0, 3, 100)
	cost_weights = np.linspace(0, 100, 100)

	comp_stripe = np.array([-35.90911969399966*i for i in comp_weights])
	comp_qudrnt = np.array([-39.25512476486815*i for i in comp_weights])
	comp_sum = np.logaddexp2(comp_stripe, comp_qudrnt)
	p_comp_stripe = 2**(comp_stripe-comp_sum)
	p_comp_qudrnt = 2**(comp_qudrnt-comp_sum)

	cost_stripe = np.array([-4.294113843380405*i for i in cost_weights])
	cost_qudrnt = np.array([-4.200174400715539*i for i in cost_weights])
	cost_sum = np.logaddexp2(cost_stripe, cost_qudrnt)
	p_cost_stripe = 2**(cost_stripe-cost_sum)
	p_cost_qudrnt = 2**(cost_qudrnt-cost_sum)

	fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), sharey=True)

	axes[0].plot(comp_weights, p_comp_stripe, c=colors.blue, linestyle='-')
	axes[0].plot(comp_weights, p_comp_qudrnt, c=colors.blue, linestyle=':')
	axes[0].set_ylabel('Prior probability')
	axes[0].set_xlabel('Weight (w)')
	axes[0].set_title('Simplicity prior (πsim)', fontsize=10)
	axes[0].set_xlim(0, comp_weights[-1])
	axes[0].text(comp_weights[-1]*0.97, 0.9, 'stripes', horizontalalignment='right', verticalalignment='center')
	axes[0].text(comp_weights[-1]*0.97, 0.1, 'quadrants', horizontalalignment='right', verticalalignment='center')

	axes[1].plot(cost_weights, p_cost_stripe, c=colors.red, linestyle='-')
	axes[1].plot(cost_weights, p_cost_qudrnt, c=colors.red, linestyle=':')
	axes[1].set_xlabel('Weight (w)')
	axes[1].set_title('Informativeness prior (πinf)', fontsize=10)
	axes[1].set_xlim(0, cost_weights[-1])
	axes[1].text(cost_weights[-1]*0.97, 0.9, 'quadrants', horizontalalignment='right', verticalalignment='center')
	axes[1].text(cost_weights[-1]*0.97, 0.1, 'stripes', horizontalalignment='right', verticalalignment='center')

	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5)
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

make_fig('../manuscript/figs/exp1_prior_comparison.eps')
