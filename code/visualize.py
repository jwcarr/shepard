'''
Module for creating visualizations of language partitions. Pass a
language array to the visualize function and specify a filename
(supports: pdf, eps, svg, and png):

visualize(language, 'some_file.pdf', show_stimuli=True)
'''

import numpy as np
import colors
import tools

radiuses = [25, 50, 75, 100, 125, 150, 175, 200]
angles = [2.5656, 3.0144, 3.4632, 3.912, 4.3608, 4.8096, 5.2583, 5.7072]

veridical_systems = {
	'angle': np.array([[i]*8 for i in [0,0,1,1,2,2,3,3]], dtype=int),
	'size' : np.array([[0,0,1,1,2,2,3,3] for _ in range(8)], dtype=int),
	'both' : np.array([[0,0,0,0,1,1,1,1] for _ in range(4)] + [[2,2,2,2,3,3,3,3] for _ in range(4)], dtype=int)
}

def create_production_svg(data, show_stimuli=True, offset_x=0, offset_y=0):
	svg = '<g id="partition">\n\n'
	for stim_i, ((y, x), category) in enumerate(np.ndenumerate(data)):
		radius, angle = radiuses[x], angles[y]
		loc_x, loc_y = x * 500 + 250 + offset_x, (y + 1) * 500 - 250 + offset_y
		box_x, box_y = x * 500 + offset_x, y * 500 + offset_y
		line_x, line_y = radius * np.cos(angle) + loc_x, radius * np.sin(angle) + loc_y
		svg += '	<g id="stimulus-%i">\n' % stim_i
		if category >= 0 and category < len(colors.categories):
			if rectangles:
				svg += '		<polygon points="%i,%i %i,%i %i,%i %i,%i" style="stroke:white; stroke-width:0; fill:%s;" />\n' % (box_x, box_y, box_x+500, box_y, box_x+500, box_y+500, box_x, box_y+500, colors.categories[category])
			else:
				svg += '		<polygon points="%i,%i %i,%i %i,%i %i,%i" style="stroke:white; stroke-width:10; fill:%s;" />\n' % (box_x, box_y, box_x+500, box_y, box_x+500, box_y+500, box_x, box_y+500, colors.categories[category])
		if show_stimuli:
			svg += '		<circle cx="%i" cy="%i" r="%i" style="stroke:black; stroke-width:10; fill:none;" />\n' % (loc_x, loc_y, radius)
			svg += '		<line x1="%i" y1="%i" x2="%f" y2="%f" style="stroke: black; stroke-width:10;" />\n' % (loc_x, loc_y, line_x, line_y)
		svg += '	</g>\n\n'
	svg += '</g>\n\n'
	return svg	

def create_production_svg_rect(data, show_stimuli=True, offset_x=0, offset_y=0):
	import rectlang
	rectlang_space = rectlang.Space((8,8), solutions_file='../data/8x8_solutions.json')
	svg = '<g id="partition">\n\n'
	for cat_i in np.unique(data):
		color = colors.categories[cat_i]
		cat_rects = rectlang_space.compress_concept(data==cat_i)[1]
		for (y, x), _, _, _, (h, w) in cat_rects:
			box_x = offset_x + (x * 500)
			box_y = offset_y + (y * 500)
			box_w = w * 500
			box_h = h * 500
			svg += "\t\t\t<rect x='%s' y='%s' width='%s' height='%s' style='fill:%s; stroke-width:0.1; stroke:%s' />\n" % (str(box_x), str(box_y), str(box_w), str(box_h), color, color)

	if show_stimuli:
		for stim_i, (y, x) in enumerate(np.ndindex(data.shape)):
			radius, angle = radiuses[x], angles[y]
			loc_x, loc_y = x * 500 + 250 + offset_x, (y + 1) * 500 - 250 + offset_y
			box_x, box_y = x * 500 + offset_x, y * 500 + offset_y
			line_x, line_y = radius * np.cos(angle) + loc_x, radius * np.sin(angle) + loc_y
			svg += '\t<g id="stimulus-%i">\n' % stim_i
			svg += '\t\t<circle cx="%i" cy="%i" r="%i" style="stroke:black; stroke-width:10; fill:none;" />\n' % (loc_x, loc_y, radius)
			svg += '\t\t<line x1="%i" y1="%i" x2="%f" y2="%f" style="stroke: black; stroke-width:10;" />\n' % (loc_x, loc_y, line_x, line_y)
			svg += '\t</g>\n\n'
	svg += '</g>\n\n'
	return svg	

def create_comprehension_svg(data, show_stimuli=True, offset_x=0, offset_y=0):
	svg = '<g id="partition">\n\n'
	offsets = [(100+offset_x,100+offset_y), (4400+offset_x,100+offset_y), (100+offset_x,4400+offset_y), (4400+offset_x,4400+offset_y)]
	for cat_i in range(4):
		for rad_i in range(8):
			rad = radiuses[rad_i]
			loc_x = rad_i * 500 + 250 + offsets[cat_i][0]
			box_x = rad_i * 500 + offsets[cat_i][0]
			for ang_i in range(8):
				ang = angles[ang_i]
				loc_y = (ang_i+1) * 500 - 250 + offsets[cat_i][1]
				box_y = ang_i * 500 + offsets[cat_i][1]
				line_x = rad * np.cos(ang) + loc_x
				line_y = rad * np.sin(ang) + loc_y
				color = fake_alpha(colors.categories[cat_i], data[cat_i,ang_i,rad_i])
				stim_i = rad_i*8 + ang_i
				svg += '	<g id="stimulus-%i-%i">\n' % (cat_i, stim_i)
				svg += '		<polygon points="%i,%i %i,%i %i,%i %i,%i" style="stroke: white; stroke-width:10; fill:%s;" />\n' % (box_x, box_y, box_x+500, box_y, box_x+500, box_y+500, box_x, box_y+500, color)
				if show_stimuli:
					svg += '		<circle cx="%i" cy="%i" r="%i" style="stroke:black; stroke-width: 10; fill:none;" />\n' % (loc_x, loc_y, rad)
					svg += '		<line x1="%i" y1="%i" x2="%f" y2="%f" style="stroke: black; stroke-width: 10;" />\n' % (loc_x, loc_y, line_x, line_y)
				svg += '	</g>\n\n'
		box1 = offsets[cat_i][0], offsets[cat_i][1]
		box2 = offsets[cat_i][0]+4000, offsets[cat_i][1]
		box3 = offsets[cat_i][0]+4000, offsets[cat_i][1]+4000
		box4 = offsets[cat_i][0], offsets[cat_i][1]+4000
		svg += '	<g id="bounding-box-%i">\n' % cat_i
		svg += '		<polygon points="%i,%i %i,%i %i,%i %i,%i" style="stroke: black; stroke-width:20; fill:none;" />\n' % (box1[0], box1[1], box2[0], box2[1], box3[0], box3[1], box4[0], box4[1])
		svg += '	</g>\n\n'
	svg += '</g>'
	return svg

def fake_alpha(hex_color, alpha): # lightens a color, producing a fake alpha transparency effect (for the sake of EPS compatibility)
	rgb_color = hex_to_rgb(hex_color)
	rgb_color = lighten(rgb_color[0], alpha), lighten(rgb_color[1], alpha), lighten(rgb_color[2], alpha)
	return rgb_to_hex(rgb_color)

def lighten(value, alpha):
	return int(round(value + ((255 - value) * (1-alpha))))

def hex_to_rgb(hex_color):
	hex_color = hex_color.lstrip('#')
	return tuple(int(hex_color[i:i+2], 16) for i in range(0, 6, 2))

def rgb_to_hex(rgb_color):
	return '#%02x%02x%02x' % tuple(map(int, map(round, rgb_color)))

def produce_categorical_veridical_systems():
	for system_name, system in veridical_systems.items():
		visualize(system, '../visuals/veridical_systems/categorical_%s.svg'%system_name)

def produce_gaussian_veridical_systems(metric='euclidean', gamma=1):
	for system_name, system in veridical_systems.items():
		partition_object = inf.partition(system)
		gaussian_distributions = []
		for category_label in range(4):
			gaussian = inf.gaussian_distribution(partition_object, category_label, metric, gamma)
			gaussian_distributions.append(gaussian / gaussian.max())
		gaussian_distributions = np.stack(gaussian_distributions, axis=2)
		visualize(gaussian_distributions, '../visuals/category_systems/gaussian_%s.pdf'%system_name)

# -------------------------------------------------------------------

def visualize(data, figure_path, figure_width=5, show_stimuli=True, rect_compress=False):
	if not isinstance(data, np.ndarray):
		raise ValueError('Input data should be numpy array')
	if len(data.shape) == 2 and (data.dtype == int or data.dtype == bool):
		height = 500 * data.shape[0]
		width = 500 * data.shape[1]
		figure_height = figure_width / width * height
		svg = '<svg width="%iin" height="%fin" viewBox="0 0 %i %i" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:svg="http://www.w3.org/2000/svg" xmlns="http://www.w3.org/2000/svg" version="1.1">\n\n' % (figure_width, figure_height, width, height)
		if rect_compress:
			svg += create_production_svg_rect(data, show_stimuli)
		else:
			svg += create_production_svg(data, show_stimuli)
	elif len(data.shape) == 3 and data.dtype == float:
		height = 500 * data.shape[1] * 2 + 500
		width = 500 * data.shape[2] * 2 + 500
		figure_height = figure_width / width * height
		svg = '<svg width="%iin" height="%fin" viewBox="0 0 %i %i" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:svg="http://www.w3.org/2000/svg" xmlns="http://www.w3.org/2000/svg" version="1.1">\n\n' % (figure_width, figure_height, width, height)
		svg += create_comprehension_svg(data)
	else:
		raise ValueError('Invalid input data. Should be 8x8 ints (production) or 4x8x8 floats (comprehension)')
	svg += '</svg>'
	with open(figure_path, mode='w', encoding='utf-8') as file:
		file.write(svg)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)

def visualize_all(data_by_participant, figure_path, figure_width=10, test_type='production', n_rows=5, n_cols=8, show_stimuli=True, label=None):
	if label:
		if test_type == 'production':
			y_offset_for_label = 2000
			label = '<text text-anchor="left" dominant-baseline="central" x="500" y="1000" fill="black" style="font-size: 1000px; font-family:Helvetica">%s</text>\n\n' % label
		else:
			y_offset_for_label = 4000
			label = '<text text-anchor="left" dominant-baseline="central" x="1000" y="2000" fill="black" style="font-size: 2000px; font-family:Helvetica">%s</text>\n\n' % label
	else:
		label = ''
	if test_type == 'production':
		height = 500 * data_by_participant[0].shape[0] * n_rows + (500*(n_rows+1)) + y_offset_for_label
		width = 500 * data_by_participant[0].shape[1] * n_cols + (500*(n_cols+1))
		figure_height = figure_width / width * height
	elif test_type == 'comprehension':
		height = (500 * data_by_participant[0].shape[1] * n_rows + (500*(n_rows+1))) * 2 + 500 + y_offset_for_label
		width = (500 * data_by_participant[0].shape[2] * n_cols + (500*(n_cols+1))) * 2 + 500
		figure_height = figure_width / width * height
	svg = '<svg width="%iin" height="%fin" viewBox="0 0 %i %i" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:svg="http://www.w3.org/2000/svg" xmlns="http://www.w3.org/2000/svg" version="1.1">\n\n%s' % (figure_width, figure_height, width, height, label)
	arrangement_iterator = np.ndindex((n_rows, n_cols))
	for partition in data_by_participant:
		position = arrangement_iterator.next()
		if test_type == 'production':
			offset_x, offset_y = position[1] * 4500 + 500, position[0] * 4500 + 500 + y_offset_for_label
			svg += create_production_svg(partition, show_stimuli=show_stimuli, offset_x=offset_x, offset_y=offset_y)
		elif test_type == 'comprehension':
			offset_x, offset_y = position[1] * 9000 + 1000, position[0] * 9000 + 1000 + y_offset_for_label
			svg += create_comprehension_svg(partition, show_stimuli=show_stimuli, offset_x=offset_x, offset_y=offset_y)
	svg += '</svg>'
	with open(figure_path, mode='w', encoding='utf-8') as file:
		file.write(svg)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)
