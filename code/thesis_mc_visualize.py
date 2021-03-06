'''
Code for visualizing mcmc chains used to sample from the hypothesis
space. This was not included in the paper; refer to my thesis instead.
'''

from os import path, remove
import numpy as np
import rectlang
import colors
import tools

radiuses = [25, 50, 75, 100, 125, 150, 175, 200]
angles = [2.5656, 3.0144, 3.4632, 3.912, 4.3608, 4.8096, 5.2583, 5.7072]

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
scale_factor = 16
figure_width = 5.5 #inches

rectlang_space = rectlang.Space((4,4))

def draw_language(language, offset_x, offset_y, chain, generation, show_stimuli=False, rect_compress=True, accepted=True):
	if rect_compress:
		return draw_language_rects(language, offset_x, offset_y, chain, generation, show_stimuli, accepted)
	return draw_language_cells(language, offset_x, offset_y, chain, generation, show_stimuli, accepted)

def draw_language_cells(language, offset_x, offset_y, chain, generation, show_stimuli=True):
	language_id = letters[chain] + str(generation)
	svg = '		<g id="language-%s">\n' % language_id
	for x in range(4):
		rad = radiuses[x] / scale_factor
		loc_x = (offset_x + (x * 1000) + 250) / scale_factor
		box_x = (offset_x + (x * 1000)) / scale_factor
		for y in range(4):
			stim_i = x*4 + y
			svg += '			<g id="stim-%s">\n' % stim_i
			ang = angles[y]
			loc_y = (offset_y + ((y+1) * 1000) - 250) / scale_factor
			box_y = (offset_y + (y * 1000)) / scale_factor
			line_x = rad * np.cos(ang) + loc_x
			line_y = rad * np.sin(ang) + loc_y
			color = colors.categories[language[y,x]]
			if generation % 2 == 1:
				color = 'black'
			svg += "				<polygon points='%s,%s %s,%s %s,%s %s,%s' style='stroke: %s; stroke-width:1; fill:%s;' />\n" % (str(box_x), str(box_y), str(box_x+(1000//scale_factor)), str(box_y), str(box_x+(1000//scale_factor)), str(box_y+(1000//scale_factor)), str(box_x), str(box_y+(1000//scale_factor)), color, color)
			if show_stimuli:
				svg += "				<circle cx='%s' cy='%s' r='%s' style='stroke:black; stroke-width:1; fill:none;' />\n" % (str(loc_x), str(loc_y), str(rad))
				svg += "				<line x1='%s' y1='%s' x2='%s' y2='%s' style='stroke: black; stroke-width:1;' />\n" % (loc_x, loc_y, line_x, line_y)
			svg += '			</g>\n'
	svg += '		</g>\n\n'
	return svg

def draw_language_rects(language, offset_x, offset_y, chain, generation, show_stimuli=True, accepted=True):
	language_id = letters[chain] + str(generation)
	svg = '\t\t<g id="language-%s">\n' % language_id
	for cat_i in np.unique(language):
		color = colors.categories[cat_i]
		if generation % 2 == 0:
			color = colors.categories[cat_i]
		else:
			color = colors.categories_out[cat_i]
		cat_rects = rectlang_space.compress_concept(language==cat_i)[1]
		for (y, x), _, _, _, (h, w) in cat_rects:
			box_x = (offset_x + (x * 1000)) / scale_factor
			box_y = (offset_y + (y * 1000)) / scale_factor
			box_w = w * 1000 / scale_factor
			box_h = h * 1000 / scale_factor
			svg += "\t\t\t<rect x='%s' y='%s' width='%s' height='%s' style='fill:%s; stroke-width:0.1; stroke:%s' />\n" % (str(box_x), str(box_y), str(box_w), str(box_h), color, color)
	if generation % 2 == 1:
		if accepted:
			svg += '\t\t\t<text text-anchor="middle" dominant-baseline="central" x="%s" y="%s" fill="black" style="font-size: %ipx; font-family:MS Mincho">✔︎</text>\n' % ((offset_x+2000)//scale_factor, ((offset_y+2000)+700)//scale_factor, 2000//scale_factor)
		else:
			svg += '\t\t\t<text text-anchor="middle" dominant-baseline="central" x="%s" y="%s" fill="black" style="font-size: %ipx; font-family:MS Mincho">✘</text>\n' % ((offset_x+2000)//scale_factor, ((offset_y+2000)+700)//scale_factor, 2000//scale_factor)

	svg += '\t\t</g>\n\n'
	return svg

def draw_letter(letter_i, offset_x, offset_y):
	letter = letters[letter_i]
	loc_x = (offset_x + (4000 / 2)) / scale_factor
	loc_y = (offset_y + (4000 / 2)) / scale_factor
	svg =  '		<g id="chain-letter-%s">\n' % letter
	svg += '			<text text-anchor="middle" dominant-baseline="central" x="%s" y="%s" fill="black" style="font-size: %ipx; font-family:Helvetica">%s</text>\n' % (loc_x, loc_y, 2000//scale_factor, letter)
	svg += '		</g>\n\n'
	return svg

def draw_number(number, offset_x, offset_y):
	loc_x = (offset_x + (4000 / 2)) / scale_factor
	loc_y = (offset_y + (4000 / 2)) / scale_factor
	svg =  '	<g id="generation-number-%i">\n' % number
	svg += '		<text text-anchor="middle" dominant-baseline="central" x="%s" y="%s" fill="black" style="font-size: %ipx; font-family:Helvetica">%s</text>\n' % (loc_x, loc_y, 2000//scale_factor, str(number))
	svg += '	</g>\n\n'
	return svg

def draw_dots(offset_x, offset_y):
	mid_loc_x= (offset_x + (4000 / 2)) / scale_factor
	mid_loc_y = (offset_y + (4000 / 2)) / scale_factor
	left_loc_x = mid_loc_x - (680//scale_factor)
	right_loc_x = mid_loc_x + (680//scale_factor)
	svg =  '		<g id="dots">\n'
	svg += "			<circle cx='%s' cy='%s' r='%i' style='stroke:black; stroke-width: 5; fill:black;' />\n" % (str(mid_loc_x), str(mid_loc_y), 200//scale_factor)
	svg += "			<circle cx='%s' cy='%s' r='%i' style='stroke:black; stroke-width: 5; fill:black;' />\n" % (str(left_loc_x), str(mid_loc_y), 200//scale_factor)
	svg += "			<circle cx='%s' cy='%s' r='%i' style='stroke:black; stroke-width: 5; fill:black;' />\n" % (str(right_loc_x), str(mid_loc_y), 200//scale_factor)
	svg += '		</g>\n\n'
	return svg

def draw_all_chains(chain_data, n_columns=10, show_stimuli=False, show_generation_numbers=False, method='productions', rect_compress=True, verbose=False):
	arr = []
	svg = ''
	offset_x = 4400
	offset_y = 400
	if show_generation_numbers:
		for gen_i in range(n_columns+1):
			svg += draw_number(gen_i, offset_x, offset_y)
			offset_x += 4400
		offset_y += 4000
	for chain_i in range(len(chain_data)):
		svg +=  '	<g id="chain-%i">\n\n' % chain_i
		chain = chain_data[chain_i]
		n_generations = len(chain['generations'])-1 # doesn't include gen 0
		# first_fixation = chain['first_fixation']
		n_full_rows = n_generations // n_columns # number of full rows that will be required
		final_row_length = n_generations % n_columns # number of gens in the final row
		if final_row_length == 0:
			n_rows = n_full_rows
		else:
			n_rows = n_full_rows + 1
		for row_i in range(n_rows):
			arr.append([])
			offset_x = 0
			for col_i in range(n_columns+2):
				if row_i == 0 and col_i == 0:
					# insert the chain number
					arr[-1].append(str(chain['chain_id']))
					svg += draw_letter(chain['chain_id'], offset_x, offset_y)
				elif row_i == 0 and col_i == 1:
					language = chain['generations'][0]['language']
					svg += draw_language(language, offset_x, offset_y, chain['chain_id'], 0, show_stimuli, rect_compress)
					arr[-1].append('0')
				elif row_i > 0 and col_i == 0:
					# blank
					arr[-1].append('-')
				elif row_i > 0 and col_i == 1:
					# insert ...
					arr[-1].append('.')
					svg += draw_dots(offset_x, offset_y)
				elif row_i >= 0 and row_i < n_rows-1 and col_i == n_columns+2:
					# insert ...
					arr[-1].append('.')
				elif row_i < n_rows-1 and col_i == n_columns+3:
					arr[-1].append('--')
				else:
					generation = (row_i * n_columns) + (col_i - 1)
					if generation <= n_generations:
						str_gen = str(generation)
						if len(str_gen) == 1:
							str_gen = '0' + str_gen
						arr[-1].append(str_gen)
						language = chain['generations'][generation]['language']
						accepted = chain['generations'][generation]['accepted']
						svg += draw_language(language, offset_x, offset_y, chain['chain_id'], generation, show_stimuli, rect_compress, accepted)
					else:
						arr[-1].append('--')
				offset_x += 4400
			offset_y += 4400
		svg += '	</g>\n\n'
	width, height = offset_x//scale_factor, offset_y//scale_factor
	figure_height = (height/width) * figure_width
	final_svg = "<svg width='%fin' height='%fin' viewBox='0 0 %i %i' xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#' xmlns:svg='http://www.w3.org/2000/svg' xmlns='http://www.w3.org/2000/svg' version='1.1'>\n\n" % (figure_width, figure_height, width, height)
	final_svg += '	<g id="background">\n'
	final_svg += '		<polygon points="0,0 %i,0 %i,%i 0,%i" style="fill:white;" />\n' % (width, width, height, height)
	final_svg += '	</g>\n\n'
	final_svg += svg
	final_svg += '</svg>'
	if verbose:
		for line in arr:
			print(line)
	return final_svg

def make_figure(data, figure_path, start_gen=0, end_gen=100, n_columns=10, show_stimuli=False, show_generation_numbers=False, method='productions', rect_compress=True, verbose=False):
	'''
	Make a figure depeciting the evolution of a bunch of chains.
	'''
	for chain in data['chains']:
		chain['generations'] = [generation for gen_i, generation in enumerate(chain['generations']) if gen_i >= start_gen and gen_i <= end_gen]

	svg = draw_all_chains(data['chains'], n_columns, show_stimuli, show_generation_numbers, method, rect_compress, verbose)

	with open(figure_path, mode='w', encoding='utf-8') as file:
		file.write(svg)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)
