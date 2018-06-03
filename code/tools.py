from os import path, remove, walk
from subprocess import call, STDOUT, DEVNULL
import re
import json

def iter_directory(directory):
	for root, dirs, files in walk(directory, topdown=False):
		for file in files:
			if file[0] != '.':
				yield path.join(root, file), file

def read_json_file(file_path):
	with open(file_path, mode='r') as file_handle:
		data = json.loads(file_handle.read())
	return data

def read_json_lines(file_path):
	with open(file_path, mode='r', encoding='utf-8') as file:
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
	svg = svg.replace('ε', '<tspan style="font-style: italic;">ε</tspan>')
	svg = svg.replace('ξ', '<tspan style="font-style: italic;">ξ</tspan>')
	svg = svg.replace('b =', '<tspan style="font-style: italic;">b</tspan> =')
	svg = svg.replace('w =', '<tspan style="font-style: italic;">w</tspan> =')
	svg = svg.replace('(w)', '(<tspan style="font-style: italic;">w</tspan>)')
	with open(svg_file_path, mode='w', encoding='utf-8') as file:
		file.write(svg)

def convert_svg(svg_file_path, file_path, remove_svg_file=False, png_width=1000):
	filename, extension = path.splitext(file_path)
	if extension not in ['.pdf', '.eps', '.png']:
		raise ValueError('Invalid format. Use either .pdf, .eps, or .png')
	if extension == '.pdf':
		call(['/usr/local/bin/inkscape', svg_file_path, '-A', file_path, '--export-text-to-path'], stdout=DEVNULL, stderr=STDOUT)
	elif extension == '.eps':
		call(['/usr/local/bin/inkscape', svg_file_path, '-E', file_path, '--export-text-to-path'], stdout=DEVNULL, stderr=STDOUT)
	elif extension == '.png':
		call(['/usr/local/bin/inkscape', svg_file_path, '-e', file_path, '--export-width=%i'%png_width], stdout=DEVNULL, stderr=STDOUT)
	if remove_svg_file:
		remove(svg_file_path)
	print('File saved to: ' + filename + extension)
