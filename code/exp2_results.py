import il_results
import il_visualize
import tools

def make_experiment_results_figure(file_path):
	experiment_results = il_results.load('../data/experiments/exp2_chains.json', start_gen=0, end_gen=10, method='prod')
	il_results.make_figure([(experiment_results, 'Experiment', '#404040', '#DADADA', '-')],
		        file_path=file_path, show_legend=False)

def make_model_comparison_figure(file_path):
	experiment_results = il_results.load('../data/experiments/exp2_chains.json', start_gen=0, end_gen=10, method='prod')
	experiment_results_est = il_results.load('../data/experiments/exp2_chains.json', start_gen=10, end_gen=50, method='prod')
	model_results_sim = il_results.load('../data/model_sim/1.36_0.23_2_4.json', start_gen=0, end_gen=50, method='lang')
	model_results_inf = il_results.load('../data/model_inf/243.3_0.37_2_4.json', start_gen=0, end_gen=50, method='lang')
	il_results.make_figure([(experiment_results, 'Experiment', '#404040', '#DADADA', '-'),
	             (experiment_results_est, '', '#404040', '#DADADA', ':'),
	             (model_results_sim, 'Model fit (πsim)', '#4D5C83', '#CACFDA', '-'),
	             (model_results_inf, 'Model fit (πinf)', '#F56A4F', '#FCD3CB', '-')],
	             file_path=file_path, show_legend=True)

def make_experiment_chains_figure(file_path):
	data = tools.read_json_file('../data/experiments/exp2_chains.json')
	il_visualize.make_figure(data, file_path, start_gen=0, end_gen=50, n_columns=15, overwrite=True)

######################################################################

# make_experiment_results_figure('../manuscript/figs/exp2_results.eps')
# make_model_comparison_figure('../manuscript/figs/exp2_model_comparison.eps')
# make_experiment_chains_figure('../manuscript/figs/exp2_chains.eps')

# make_experiment_results_figure('../visuals/exp2_results.pdf')
# make_model_comparison_figure('../visuals/exp2_model_comparison.pdf')
# make_experiment_chains_figure('../visuals/exp2_chains.pdf')
