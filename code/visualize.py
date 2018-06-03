'''
Module for creating visualizations of language partitions. Pass a
language array to the visualize function and specify a filename
(supports: pdf, eps, svg, and png):

visualize(language, 'some_file.pdf', show_stimuli=True)
'''

from os import path, remove
import numpy as np
import tools

radiuses = [25, 50, 75, 100, 125, 150, 175, 200]
angles = [2.5656, 3.0144, 3.4632, 3.912, 4.3608, 4.8096, 5.2583, 5.7072]
category_colors = ['#E85A71', '#6B6B7F', '#4EA1D3', '#FCBE32', '#FFFFFF']

veridical_systems = {
	'angle': np.array([[i]*8 for i in [0,0,1,1,2,2,3,3]], dtype=int),
	'size' : np.array([[0,0,1,1,2,2,3,3] for _ in range(8)], dtype=int),
	'both' : np.array([[0,0,0,0,1,1,1,1] for _ in range(4)] + [[2,2,2,2,3,3,3,3] for _ in range(4)], dtype=int)
}

def create_production_svg(data, show_stimuli=True, show_labels=False, rectangles=None, uncomp_recs=None):
	if show_labels:
		svg = "<svg width='4500' height='4200' xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#' xmlns:svg='http://www.w3.org/2000/svg' xmlns='http://www.w3.org/2000/svg' version='1.1'>\n\n"
	else:
		svg = "<svg width='4000' height='4000' xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#' xmlns:svg='http://www.w3.org/2000/svg' xmlns='http://www.w3.org/2000/svg' version='1.1'>\n\n"
	for rad_i in range(8):
		rad = radiuses[rad_i]
		if show_labels:
			svg += '	<g id="radius-label-%i">\n' % rad_i
			svg += '		<text text-anchor="middle" x="%s" y="4150" fill="black" style="font-size: 75px; font-family:Helvetica;">%s px</text>\n' % (rad_i*500+750, str(radiuses[rad_i]))
			svg += '	</g>\n\n'
			loc_x = (rad_i+1) * 500 + 250
			box_x = (rad_i+1) * 500
		else:
			loc_x = rad_i * 500 + 250
			box_x = rad_i * 500
		for ang_i in range(8):
			if show_labels and rad_i == 0:
				ang_deg = round((angles[ang_i] / (np.pi*2)) * 360, 2)
				ang_rad = round(angles[ang_i], 2)
				svg += '	<g id="angle-label-%i">\n' % ang_i
				svg += '		<text text-anchor="middle" x="250" y="%s" fill="black" style="font-size: 75px; font-family:Helvetica;">%s°</text>\n' % (ang_i*500+250, str(ang_deg))
				svg += '		<text text-anchor="middle" x="250" y="%s" fill="black" style="font-size: 60px; font-family:Helvetica;">%s rad</text>\n' % (ang_i*500+350, str(ang_rad))
				svg += '	</g>\n\n'
			ang = angles[ang_i]
			loc_y = (ang_i+1) * 500 - 250
			box_y = ang_i * 500
			line_x = rad * np.cos(ang) + loc_x
			line_y = rad * np.sin(ang) + loc_y
			color = category_colors[data[ang_i,rad_i]]
			stim_i = rad_i*8 + ang_i
			svg += '	<g id="stimulus-%i">\n' % stim_i
			svg += '		<polygon points="%i,%i %i,%i %i,%i %i,%i" style="stroke: white; stroke-width:10; fill:%s;" />\n' % (box_x, box_y, box_x+500, box_y, box_x+500, box_y+500, box_x, box_y+500, color)
			if show_stimuli:
				svg += '		<circle cx="%i" cy="%i" r="%i" style="stroke:black; stroke-width: 10; fill:none;" />\n' % (loc_x, loc_y, rad)
				svg += '		<line x1="%i" y1="%i" x2="%f" y2="%f" style="stroke: black; stroke-width: 10;" />\n' % (loc_x, loc_y, line_x, line_y)
			svg += '	</g>\n\n'
	if rectangles:
		for category_rectangles in rectangles:
			for rectangle in category_rectangles:
				topleft     = rectangle[0][0]*500, rectangle[0][1]*500
				topright    = rectangle[1][0]*500, rectangle[1][1]*500
				bottomleft  = rectangle[2][0]*500, rectangle[2][1]*500
				bottomright = rectangle[3][0]*500, rectangle[3][1]*500
				svg += '	<g id="rectangle">\n'
				svg += '		<polygon points="%i,%i %i,%i %i,%i %i,%i" style="stroke: white; stroke-width:30; fill:none;" />\n' % (topleft[0], topleft[1], topright[0], topright[1], bottomright[0], bottomright[1], bottomleft[0], bottomleft[1])
				svg += '	</g>\n\n'
	svg += '</svg>'
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
				color = fake_alpha(category_colors[cat_i], data[ang_i,rad_i,cat_i])
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

# -------------------------------------------------------------------

def visualize(data, filename, overwrite=False, show_stimuli=True, show_labels=False, rectangles=None, uncomp_recs=False):
	if not isinstance(data, np.ndarray):
		raise ValueError('Input data should be numpy array')
	if path.isfile(filename) and not overwrite:
		raise ValueError('Could not write to path: ' + str(filename) + '. Set overwrite=True to override')
	if len(data.shape) == 2 and (data.dtype == int or data.dtype == bool):
		svg = create_production_svg(data, show_stimuli, show_labels, rectangles, uncomp_recs)
	elif len(data.shape) == 3 and data.dtype == float:
		svg = create_comprehension_svg(data)
	else:
		raise ValueError('Invalid input data. Should be 8x8 ints (production) or 8x8x4 floats (comprehension)')
	with open(filename, mode='w', encoding='utf-8') as file:
		file.write(svg)
	if not filename.endswith('.svg'):
		tools.convert_svg(filename, filename)

# -------------------------------------------------------------------

def produce_categorical_veridical_systems():
	for system_name, system in veridical_systems.items():
		visualize(system, '../visuals/veridical_systems/categorical_%s.svg'%system_name, overwrite=True)

def produce_gaussian_veridical_systems(metric='euclidean', gamma=1):
	for system_name, system in veridical_systems.items():
		partition_object = inf.partition(system)
		gaussian_distributions = []
		for category_label in range(4):
			gaussian = inf.gaussian_distribution(partition_object, category_label, metric, gamma)
			gaussian_distributions.append(gaussian / gaussian.max())
		gaussian_distributions = np.stack(gaussian_distributions, axis=2)
		visualize(gaussian_distributions, '../visuals/category_systems/gaussian_%s.pdf'%system_name, overwrite=True)
