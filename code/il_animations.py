import numpy as np
import imageio

colors = [[232,90,113], [107,107,127], [78,161,211], [252,190,50], [255,255,255]]
colors = [np.array(color, dtype=np.uint8) for color in colors]

def make_image(language, cell_width=16, grid_width=1):
	width = language.shape[0] * cell_width - grid_width
	height = language.shape[0] * cell_width - grid_width
	image = np.full((width,height,3), 255, dtype=np.uint8)
	for (x, y), category in np.ndenumerate(language):
		image[x*cell_width:((x+1)*cell_width)-grid_width,
			  y*cell_width:((y+1)*cell_width)-grid_width] = colors[category]
	return image

def save_image(language, output_file, cell_width=16, grid_width=1):
	image = make_image(language, cell_width, grid_width)
	imageio.imsave(output_file, image)

def save_animation(best_chain, out_file, show_seen=False, create_thumbnail=False):
	generations = best_chain['generations']
	images = []
	if create_thumbnail:
		language = np.array(generations[-1]['language'], dtype=int).reshape((8,8))
		save_image(language, out_file + '_thumb.gif')
	for gen_i, generation in enumerate(generations):
		language = np.array(generation['language'], dtype=int).reshape((8,8))
		images.append(make_image(language))
		if show_seen:
			training = np.full((8,8), 4, dtype=int)
			for meaning, signal in generation['data_out']:
				training[tuple(meaning)] = signal
			images.append(make_image(training))
	if show_seen:
		imageio.mimsave(out_file + '.gif', images[:-1], fps=2, loop=1, palettesize=8, subrectangles=True)
	else:
		imageio.mimsave(out_file + '.gif', images, fps=2, loop=1, palettesize=8, subrectangles=True)
