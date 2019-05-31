import il_results
import il_visualize
import il_animations
import colors
import tools

def web_model_results(dir_path):
	for noise in ['0.01', '0.05', '0.1']:
		for bottleneck in ['1', '2', '3', '4']:
			for exposures in ['1', '2', '3', '4']:
				figure_path = dir_path + '%s_%s_%s.svg' % (noise, bottleneck, exposures)
				model_results_sim = il_results.load('../data/model_sim/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				model_results_inf = il_results.load('../data/model_inf/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				model_results_inf_500 = il_results.load('../data/model_inf/500.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				il_results.make_figure([(model_results_sim, 'Simplicity', colors.blue, colors.light_blue, '-'),
				                       (model_results_inf, 'Informativeness', colors.red, colors.light_red, '-'),
				                       (model_results_inf_500, 'Strong informativeness', colors.red, colors.light_red, ':')],
				                       figure_path=figure_path, show_legend=True)

def supplementary_model_results(dir_path):
	for e, noise in enumerate(['0.01', '0.05', '0.1'], 1):
		for b, bottleneck in enumerate(['1', '2', '3', '4'], 1):
			for x, exposures in enumerate(['1', '2', '3', '4'], 1):
				figure_path = dir_path + '%i%i%i.pdf' % (b, x, e)
				title = 'b = %s, ξ = %s, ε = %s' % (bottleneck, exposures, noise)
				model_results_sim = il_results.load('../data/model_sim/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				model_results_inf = il_results.load('../data/model_inf/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				model_results_inf_500 = il_results.load('../data/model_inf/500.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				il_results.make_figure([(model_results_sim, 'Simplicity prior (πsim, w = 1)', colors.blue, colors.light_blue, '-'),
				                       (model_results_inf, 'Informativeness prior (πinf, w = 1)', colors.red, colors.light_red, '-'),
				                       (model_results_inf_500, 'Strong informativeness prior (πinf, w = 500)', colors.red, colors.light_red, ':')],
				                       figure_path=figure_path, title=title, show_legend=True, deep_legend=True)

def thesis_appendix_model_results(dir_path):	
	for b, bottleneck in enumerate(['1', '2', '3', '4'], 1):
		for x, exposures in enumerate(['1', '2', '3', '4'], 1):
			for e, noise in enumerate(['0.01', '0.05', '0.1'], 1):
				figure_path = dir_path + '%i%i%i.eps' % (b, x, e)
				title = 'b = %s, ξ = %s, ε = %s' % (bottleneck, exposures, noise)
				model_results_sim = il_results.load('../data/model_sim/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				model_results_inf = il_results.load('../data/model_inf/1.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				model_results_inf_500 = il_results.load('../data/model_inf/500.0_%s_%s_%s.json' % (noise, bottleneck, exposures), start_gen=0, end_gen=50, method='lang')
				il_results.make_figure([(model_results_sim, 'Simplicity prior (πsim, w = 1)', colors.blue, colors.light_blue, '-'),
				                       (model_results_inf, 'Informativeness prior (πinf, w = 1)', colors.red, colors.light_red, '-'),
				                       (model_results_inf_500, 'Strong informativeness prior (πinf, w = 500)', colors.red, colors.light_red, ':')],
				                       figure_path=figure_path, title=title, show_legend=False, figsize=(5.1, 2.8))

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

def make_model_results_figure(figure_path):
	model_results_sim = il_results.load('../data/model_sim/1.0_0.01_2_2.json', start_gen=0, end_gen=50, method='lang')
	model_results_inf = il_results.load('../data/model_inf/1.0_0.01_2_2.json', start_gen=0, end_gen=50, method='lang')
	model_results_inf_500 = il_results.load('../data/model_inf/500.0_0.01_2_2.json', start_gen=0, end_gen=50, method='lang')
	il_results.make_figure([(model_results_sim, 'Simplicity prior', colors.blue, colors.light_blue, '-'),
	                        (model_results_inf, 'Informativeness prior', colors.red, colors.light_red, '-'),
	                        (model_results_inf_500, 'Strong informativeness prior', colors.red, colors.light_red, ':')],
	                        figure_path=figure_path, show_legend=True)

def make_model_results_figure_carstensen(figure_path):
	model_results_sim_001 = il_results.load('../data/model_sim/1.0_0.01_4_2.json', start_gen=0, end_gen=100, method='lang')
	model_results_sim_005 = il_results.load('../data/model_sim/1.0_0.05_4_2.json', start_gen=0, end_gen=100, method='lang')
	model_results_sim_010 = il_results.load('../data/model_sim/1.0_0.1_4_2.json', start_gen=0, end_gen=100, method='lang')
	il_results.make_carstensen_figure([(model_results_sim_001, 'ε = .01', colors.blue, colors.light_blue, '-'),
	                                   (model_results_sim_005, 'ε = .05', colors.blue, colors.light_blue, '--'),
	                                   (model_results_sim_010, 'ε = .1', colors.blue, colors.light_blue, ':')],
	                                   figure_path=figure_path, figsize=(3.54, 1.75))

def make_model_chains_figure(figure_path):
	best_chain_sim = il_results.extract_dataset(tools.read_json_file('../data/model_sim/1.0_0.01_2_2.json'), 0, 50, 'lang_cost', True)
	best_chain_inf = il_results.extract_dataset(tools.read_json_file('../data/model_inf/1.0_0.01_2_2.json'), 0, 50, 'lang_cost', True)
	best_chain_strong_inf = il_results.extract_dataset(tools.read_json_file('../data/model_inf/500.0_0.01_2_2.json'), 0, 50, 'lang_cost', True)
	best_chain_sim['chain_id'] = 0
	best_chain_inf['chain_id'] = 1
	best_chain_strong_inf['chain_id'] = 2
	data = {'chains':[best_chain_sim, best_chain_inf, best_chain_strong_inf]}
	il_visualize.make_figure(data, figure_path, start_gen=0, end_gen=50, n_columns=17, method='language', rect_compress=True)

def make_final_gen_dist_figure(figure_path):
	primary_results_sim = il_results.extract_generation_distribution('../data/model_sim/1.0_0.05_2_2.json', 'lang_complexity', 50)
	primary_results_inf = il_results.extract_generation_distribution('../data/model_inf/500.0_0.05_2_2.json', 'lang_cost', 50)
	final_gen_results = [
	    ('noise', [
	        ('ε = .1', colors.black, il_results.extract_generation_distribution('../data/model_sim/1.0_0.1_2_2.json', 'lang_complexity', 50)),
	        ('ε = .05', colors.blue, primary_results_sim),
	        ('ε = .01', colors.black, il_results.extract_generation_distribution('../data/model_sim/1.0_0.01_2_2.json', 'lang_complexity', 50))
	    ]),
	    ('bottleneck', [
	        ('b = 3', colors.black, il_results.extract_generation_distribution('../data/model_sim/1.0_0.05_3_2.json', 'lang_complexity', 50)),
	        ('b = 2', colors.blue, primary_results_sim),
	        ('b = 1', colors.black, il_results.extract_generation_distribution('../data/model_sim/1.0_0.05_1_2.json', 'lang_complexity', 50))
	    ]),
	    ('exposures', [
	        ('ξ = 3', colors.black, il_results.extract_generation_distribution('../data/model_sim/1.0_0.05_2_3.json', 'lang_complexity', 50)),
	        ('ξ = 2', colors.blue, primary_results_sim),
	        ('ξ = 1', colors.black, il_results.extract_generation_distribution('../data/model_sim/1.0_0.05_2_1.json', 'lang_complexity', 50))
	    ]),
	    ('noise', [
	        ('ε = .1', colors.black, il_results.extract_generation_distribution('../data/model_inf/500.0_0.1_2_2.json', 'lang_cost', 50)),
	        ('ε = .05', colors.red, primary_results_inf),
	        ('ε = .01', colors.black, il_results.extract_generation_distribution('../data/model_inf/500.0_0.01_2_2.json', 'lang_cost', 50))
	    ]),
	    ('bottleneck', [
	        ('b = 3', colors.black, il_results.extract_generation_distribution('../data/model_inf/500.0_0.05_3_2.json', 'lang_cost', 50)),
	        ('b = 2', colors.red, primary_results_inf),
	        ('b = 1', colors.black, il_results.extract_generation_distribution('../data/model_inf/500.0_0.05_1_2.json', 'lang_cost', 50))
	    ]),
	    ('exposures', [
	        ('ξ = 3', colors.black, il_results.extract_generation_distribution('../data/model_inf/500.0_0.05_2_3.json', 'lang_cost', 50)),
	        ('ξ = 2', colors.red, primary_results_inf),
	        ('ξ = 1', colors.black, il_results.extract_generation_distribution('../data/model_inf/500.0_0.05_2_1.json', 'lang_cost', 50))
	    ])
	]
	il_results.plot_final_gen_distributions(final_gen_results, figure_path)

######################################################################

# web_model_results('/Users/jon/Sites/shepard_results/figs/')
# web_animations('/Users/jon/Sites/shepard_results/anis/')

# supplementary_model_results('../visuals/model/')

# make_model_results_figure('../manuscript/figs/model_results.eps')
# make_model_chains_figure('../manuscript/figs/model_chains.eps')
# make_final_gen_dist_figure('../manuscript/figs/model_final_distributions.eps')
# make_model_results_figure_carstensen('../manuscript/figs/model_results_wide.eps')

# make_model_results_figure('../visuals/model_results.pdf')
# make_model_chains_figure('../visuals/model_chains.pdf')
# make_final_gen_dist_figure('../visuals/model_final_distributions.pdf')
# make_model_results_figure_carstensen('../visuals/model_results_wide.pdf')
