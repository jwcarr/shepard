import il_results
import il_visualize
import colors
import tools

def make_experiment_results_figure(figure_path):
	experiment_results = il_results.load('../data/experiments/exp2_chains.json', start_gen=0, end_gen=10, method='prod')
	il_results.make_figure([(experiment_results, 'Experiment', colors.black, colors.light_black, '-')],
		        figure_path=figure_path, show_legend=False)

def make_model_comparison_figure(figure_path):
	experiment_results = il_results.load('../data/experiments/exp2_chains.json', start_gen=0, end_gen=10, method='prod')
	experiment_results_est = il_results.load('../data/experiments/exp2_chains.json', start_gen=10, end_gen=50, method='prod')
	model_results_sim = il_results.load('../data/model_sim/1.36_0.23_2_4.json', start_gen=0, end_gen=50, method='lang')
	model_results_inf = il_results.load('../data/model_inf/243.3_0.37_2_4.json', start_gen=0, end_gen=50, method='lang')
	il_results.make_figure([(experiment_results, 'Experiment', colors.black, colors.light_black, '-'),
	             (experiment_results_est, '', colors.black, colors.light_black, ':'),
	             (model_results_sim, 'Model fit (πsim)', colors.blue, colors.light_blue, '-'),
	             (model_results_inf, 'Model fit (πinf)', colors.red, colors.light_red, '-')],
	             figure_path=figure_path, show_legend=True)

def make_experiment_chains_figure(figure_path):
	data = tools.read_json_file('../data/experiments/exp2_chains.json')
	il_visualize.make_figure(data, figure_path, start_gen=0, end_gen=50, n_columns=14, rect_compress=True)

######################################################################

# make_experiment_results_figure('../manuscript/figs/exp2_results.eps')
# make_model_comparison_figure('../manuscript/figs/exp2_model_comparison.eps')
# make_experiment_chains_figure('../manuscript/figs/exp2_chains.eps')

# make_experiment_results_figure('../visuals/exp2_results.pdf')
# make_model_comparison_figure('../visuals/exp2_model_comparison.pdf')
# make_experiment_chains_figure('../visuals/exp2_chains.pdf')
