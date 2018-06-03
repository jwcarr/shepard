import il_results
import il_visualize
import il_animations
import tools

def web_model_results(dir_path):
	for noise in ['0.01', '0.05', '0.1']:
		for bottleneck in ['1', '2', '3', '4']:
			for exposures in ['1', '2', '3', '4']:
				file_path = dir_path + '%s_%s_%s.svg' % (noise, bottleneck, exposures)
				model_results_sim = il_results.load('../data/model_sim/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				model_results_inf = il_results.load('../data/model_inf/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				model_results_inf_500 = il_results.load('../data/model_inf/500.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				il_results.make_figure([(model_results_sim, 'Simplicity', '#4D5C83', '#CACFDA', '-'),
				                       (model_results_inf, 'Informativeness', '#F56A4F', '#FCD3CB', '-'),
				                       (model_results_inf_500, 'Strong informativeness', '#F56A4F', '#FCD3CB', ':')],
				                       file_path=file_path, show_legend=True)

def supplementary_model_results(dir_path):
	for e, noise in enumerate(['0.01', '0.05', '0.1'], 1):
		for b, bottleneck in enumerate(['1', '2', '3', '4'], 1):
			for x, exposures in enumerate(['1', '2', '3', '4'], 1):
				file_path = dir_path + '%i%i%i.pdf' % (b, x, e)
				title = 'b = %s, ξ = %s, ε = %s' % (bottleneck, exposures, noise)
				model_results_sim = il_results.load('../data/model_sim/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				model_results_inf = il_results.load('../data/model_inf/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				model_results_inf_500 = il_results.load('../data/model_inf/500.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				il_results.make_figure([(model_results_sim, 'Simplicity prior (πsim, w = 1)', '#4D5C83', '#CACFDA', '-'),
				                       (model_results_inf, 'Informativeness prior (πinf, w = 1)', '#F56A4F', '#FCD3CB', '-'),
				                       (model_results_inf_500, 'Strong informativeness prior (πinf, w = 500)', '#F56A4F', '#FCD3CB', ':')],
				                       file_path=file_path, title=title, show_legend=True, deep_legend=True)

def web_animations(dir_path):
	for noise in ['0.01', '0.05', '0.1']:
		for bottleneck in ['1', '2', '3', '4']:
			for exposures in ['1', '2', '3', '4']:

				input_file = '../data/model_sim/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures)
				output_file = dir_path + 's_1.0_%s_%s_%s' % (noise, bottleneck, exposures)
				best_chain = il_results.extract_dataset(tools.read_json_file(input_file), 0, 50, 'lang_cost', True)
				il_animations.save_animation(best_chain, output_file, show_seen=False, create_thumbnail=True)
				il_animations.save_animation(best_chain, output_file+'_seen', show_seen=True, create_thumbnail=False)

				input_file = '../data/model_inf/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures)
				output_file = dir_path + 'i_1.0_%s_%s_%s' % (noise, bottleneck, exposures)
				best_chain = il_results.extract_dataset(tools.read_json_file(input_file), 0, 50, 'lang_cost', True)
				il_animations.save_animation(best_chain, output_file, show_seen=False, create_thumbnail=True)
				il_animations.save_animation(best_chain, output_file+'_seen', show_seen=True, create_thumbnail=False)

				input_file = '../data/model_inf/500.0_%s_%s_%s.json' % (noise, bottleneck, exposures)
				output_file = dir_path + 'i_500.0_%s_%s_%s' % (noise, bottleneck, exposures)
				best_chain = il_results.extract_dataset(tools.read_json_file(input_file), 0, 50, 'lang_cost', True)
				il_animations.save_animation(best_chain, output_file, show_seen=False, create_thumbnail=True)
				il_animations.save_animation(best_chain, output_file+'_seen', show_seen=True, create_thumbnail=False)

def make_model_results_figure(file_path):
	model_results_sim = il_results.load('../data/model_sim/1.0_0.01_2_2.json', start_gen=0, end_gen=50, method='lang')
	model_results_inf = il_results.load('../data/model_inf/1.0_0.01_2_2.json', start_gen=0, end_gen=50, method='lang')
	model_results_inf_500 = il_results.load('../data/model_inf/500.0_0.01_2_2.json', start_gen=0, end_gen=50, method='lang')
	il_results.make_figure([(model_results_sim, 'Simplicity', '#4D5C83', '#CACFDA', '-'),
	                        (model_results_inf, 'Informativeness', '#F56A4F', '#FCD3CB', '-'),
	                        (model_results_inf_500, 'Strong informativeness', '#F56A4F', '#FCD3CB', ':')],
	                        file_path=file_path, show_legend=True)

def make_wide_model_results_figure(file_path):
	model_results_sim_001 = il_results.load('../data/model_sim/1.0_0.01_4_2.json', start_gen=0, end_gen=50, method='lang')
	model_results_sim_010 = il_results.load('../data/model_sim/1.0_0.1_4_2.json', start_gen=0, end_gen=50, method='lang')
	il_results.make_figure([(model_results_sim_001, 'Simplicity prior (ε = 0.01)', '#4D5C83', '#CACFDA', '-'),
	                        (model_results_sim_010, 'Simplicity prior (ε = 0.1)', '#4D5C83', '#CACFDA', ':')],
	                        file_path=file_path, show_legend=True)

def make_model_chains_figure(file_path):
	best_chain_sim = il_results.extract_dataset(tools.read_json_file('../data/model_sim/1.0_0.01_2_2.json'), 0, 50, 'lang_cost', True)
	best_chain_inf = il_results.extract_dataset(tools.read_json_file('../data/model_inf/1.0_0.01_2_2.json'), 0, 50, 'lang_cost', True)
	best_chain_strong_inf = il_results.extract_dataset(tools.read_json_file('../data/model_inf/500.0_0.01_2_2.json'), 0, 50, 'lang_cost', True)
	best_chain_sim['chain_id'] = 0
	best_chain_inf['chain_id'] = 1
	best_chain_strong_inf['chain_id'] = 2
	data = {'chains':[best_chain_sim, best_chain_inf, best_chain_strong_inf]}
	il_visualize.make_figure(data, file_path, start_gen=0, end_gen=50, n_columns=15, method='language', overwrite=True)

# web_model_results('/Users/jon/Sites/shepard_results/figs/')
# web_animations('/Users/jon/Sites/shepard_results/anis/')

# supplementary_model_results('../visuals/model/')

# make_model_results_figure('../manuscript/figs/model_results.eps')
# make_wide_model_results_figure('../manuscript/figs/model_results_wide.eps')
# make_model_chains_figure('../manuscript/figs/model_chains.eps')

# make_model_results_figure('../visuals/model_results.pdf')
# make_wide_model_results_figure('../visuals/model_results_wide.pdf')
# make_model_chains_figure('../visuals/model_chains.pdf')
