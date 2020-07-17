from os import path, walk
import re
import json

def iter_directory(directory):
	for root, dirs, files in walk(directory, topdown=False):
		for file in files:
			if file[0] != '.':
				yield path.join(root, file), file

def read_json_file(data_path):
	with open(data_path, mode='r') as file_handle:
		data = json.loads(file_handle.read())
	return data

def read_json_lines(data_path):
	with open(data_path, mode='r', encoding='utf-8') as file:
		data = [json.loads(line) for line in file if len(line) > 1]
	return data

def format_svg_labels(svg_file_path):
	'''
	Applies some nicer formatting to an SVG plot, including setting
	the font to Helvetica and adding italics. Requires you to set
	this at the top of the script:
	plt.rcParams['svg.fonttype'] = 'none'
	'''
	with open(svg_file_path, mode='r', encoding='utf-8') as file:
		svg = file.read()
	svg = re.sub(r'font-family:.*?;', 'font-family:Helvetica Neue;', svg)
	svg = svg.replace('πsim', '<tspan style="font-style: italic;">π</tspan><tspan baseline-shift="sub" style="font-size: 6pt">sim</tspan>')
	svg = svg.replace('πinf', '<tspan style="font-style: italic;">π</tspan><tspan baseline-shift="sub" style="font-size: 6pt">inf</tspan>')
	svg = svg.replace('<g id="legend_1">', '<g id="legend_1" transform="translate(10)">')
	svg = svg.replace('γ', '<tspan style="font-style: italic;">γ</tspan>')
	svg = svg.replace('ε', '<tspan style="font-style: italic;">ε</tspan>')
	svg = svg.replace('ξ', '<tspan style="font-style: italic;">ξ</tspan>')
	svg = svg.replace('b =', '<tspan style="font-style: italic;">b</tspan> =')
	svg = svg.replace('w =', '<tspan style="font-style: italic;">w</tspan> =')
	svg = svg.replace('(w)', '(<tspan style="font-style: italic;">w</tspan>)')
	with open(svg_file_path, mode='w', encoding='utf-8') as file:
		file.write(svg)

def convert_svg(svg_file_path, out_file_path, png_width=1000):
	import cairosvg
	filename, extension = path.splitext(out_file_path)
	if extension not in ['.pdf', '.eps', '.png']:
		raise ValueError('Invalid format. Use either .pdf, .eps, or .png')
	if extension == '.pdf':
		cairosvg.svg2pdf(url=svg_file_path, write_to=out_file_path)
	elif extension == '.eps':
		cairosvg.svg2ps(url=svg_file_path, write_to=out_file_path)
	elif extension == '.png':
		cairosvg.svg2png(url=svg_file_path, write_to=out_file_path, dpi=300)
