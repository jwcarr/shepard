'''
Builds a figure depicting all rectlang symbols for a given size
universe.
'''

import numpy as np
import rectlang
import tools

cell_size = 10
symbol_padding = 12
rect_padding = 2
font_size = 10

def draw_symbol(rect, pixel_position, symbol_size, codeword):
	pixel_position = pixel_position[0]-rect_padding, pixel_position[1]-rect_padding
	symbol_size = symbol_size[0]+(2*rect_padding), symbol_size[1]+(2*rect_padding)
	rect_height, rect_width = rect[0]*cell_size, rect[1]*cell_size
	rect_y, rect_x = pixel_position[0] + (rect[2]*cell_size)+rect_padding, pixel_position[1] + (rect[3]*cell_size)+rect_padding
	svg =  "	<g id='%s'>\n" % codeword
	svg += "		<rect x='%i' y='%i' width='%i' height='%i' style='stroke:black; stroke-width:1; fill:white;' />\n" % (pixel_position[1], pixel_position[0], symbol_size[1], symbol_size[0])
	svg += "		<rect x='%i' y='%i' width='%i' height='%i' style='stroke:black; stroke-width:0; fill:black;' />\n" % (rect_x, rect_y, rect_width, rect_height)
	svg += "		<text text-anchor='middle' dominant-baseline='top' x='%s' y='%s' fill='black' style='font-size: %ipx; font-family:Menlo'>%s</text>\n" % (pixel_position[1]+symbol_size[1]/2, pixel_position[0]+symbol_size[0]+symbol_padding, font_size, codeword)
	svg += "	</g>\n\n"
	return svg

def build_figure(universe, figure_width=5, n_cols=10, n_pages=1):
	n_symbols = len(universe.symbol_to_string)
	n_rows = n_symbols // n_cols + (n_symbols % n_cols > 0)
	n_rows = n_rows // n_pages
	symbol_size = universe.shape[0]*cell_size, universe.shape[1]*cell_size
	height, width = n_rows*symbol_size[0]+((n_rows+1)*symbol_padding*2)-symbol_padding, n_cols*symbol_size[1]+((n_cols+1)*symbol_padding)
	figure_height = figure_width / width * height
	pages = []
	codewords = sorted(universe.string_to_symbol.keys())
	codeword_i = 0
	for page_i in range(n_pages):
		arrangement_iterator = np.ndindex((n_rows, n_cols))
		svg = "<svg width='%fin' height='%fin' viewBox='0 0 %i %i' xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#' xmlns:svg='http://www.w3.org/2000/svg' xmlns='http://www.w3.org/2000/svg' version='1.1'>\n\n" % (figure_width, figure_height, width, height)
		svg += "	<rect x='0' y='0' width='%i' height='%i' style='stroke-width:0; fill:white;' />\n\n" % (width, height)
		for _ in range(n_cols*n_rows):
			try:
				codeword = codewords[codeword_i]
			except IndexError:
				break
			position = arrangement_iterator.next()
			pixel_position = position[0]*symbol_size[0]+((position[0]+1)*symbol_padding*2)-symbol_padding, (position[1]*symbol_size[1]+((position[1]+1)*symbol_padding))
			rect = universe.string_to_symbol[codeword]
			svg += draw_symbol(rect, pixel_position, symbol_size, codeword)
			codeword_i += 1
		svg += "</svg>"
		pages.append(svg)
	if len(pages) == 1:
		return pages[0]
	return pages

def save_figure(figure, filename):
	with open(filename, mode='w', encoding='utf-8') as file:
		file.write(figure)
	if not filename.endswith('.svg'):
		tools.convert_svg(filename, filename)

######################################################################

# universe = rectlang.Space((4,4), max_exhaustive_size=16)
# figure = build_figure(universe, figure_width=5.5, n_cols=10)
# save_figure(figure, '../manuscript/figs/symbols.eps')

# T_shape = np.array([[1,1,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]], dtype=bool)
# print('Binary string for T-shaped region: ', universe.encode_concept(T_shape))
