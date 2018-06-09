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

def create_production_svg(data, show_stimuli=True, rectangles=None):
	height, width = 500 * data.shape[0], 500 * data.shape[1]
	svg = '<svg width="%s" height="%s" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:svg="http://www.w3.org/2000/svg" xmlns="http://www.w3.org/2000/svg" version="1.1">\n\n<g id="partition">\n\n' % (width, height)
	for stim_i, ((y, x), category) in enumerate(np.ndenumerate(data)):
		radius, angle = radiuses[x], angles[y]
		loc_x, loc_y = x * 500 + 250, (y + 1) * 500 - 250
		box_x, box_y = x * 500, y * 500
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
	if rectangles:
		for category_rectangles in rectangles:
			for rectangle in category_rectangles:
				topleft     = rectangle[0][1]*500, rectangle[0][0]*500
				topright    = rectangle[1][1]*500, rectangle[1][0]*500
				bottomleft  = rectangle[2][1]*500, rectangle[2][0]*500
				bottomright = rectangle[3][1]*500, rectangle[3][0]*500
				svg += '	<g id="rectangle">\n'
				svg += '		<polygon points="%i,%i %i,%i %i,%i %i,%i" style="stroke:white; stroke-width:30; fill:none;" />\n' % (topleft[0], topleft[1], topright[0], topright[1], bottomright[0], bottomright[1], bottomleft[0], bottomleft[1])
				svg += '	</g>\n\n'
	svg += '</g>\n\n</svg>'
	return svg	

def create_comprehension_svg(data):
	svg = "<svg width='8500' height='8500' xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#' xmlns:svg='http://www.w3.org/2000/svg' xmlns='http://www.w3.org/2000/svg' version='1.1'>\n\n"
	offsets = [(100,100), (4400,100), (100,4400), (4400,4400)]
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
				color = fake_alpha(colors.categories[cat_i], data[ang_i,rad_i,cat_i])
				stim_i = rad_i*8 + ang_i
				svg += '	<g id="stimulus-%i-%i">\n' % (cat_i, stim_i)
				svg += '		<polygon points="%i,%i %i,%i %i,%i %i,%i" style="stroke: white; stroke-width:10; fill:%s;" />\n' % (box_x, box_y, box_x+500, box_y, box_x+500, box_y+500, box_x, box_y+500, color)
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
	svg += '</svg>'
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

def visualize(data, figure_path, show_stimuli=True, rectangles=None):
	if not isinstance(data, np.ndarray):
		raise ValueError('Input data should be numpy array')
	if len(data.shape) == 2 and (data.dtype == int or data.dtype == bool):
		svg = create_production_svg(data, show_stimuli, rectangles)
	elif len(data.shape) == 3 and data.dtype == float:
		svg = create_comprehension_svg(data)
	else:
		raise ValueError('Invalid input data. Should be 8x8 ints (production) or 8x8x4 floats (comprehension)')
	with open(figure_path, mode='w', encoding='utf-8') as file:
		file.write(svg)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)
