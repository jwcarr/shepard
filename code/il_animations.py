import numpy as np
import imageio
import colors

# Convert category colors from hex triplets to RGB as 8-bit unsigned
# integer arrays (required by imageio)
category_colors = []
for hex_color in colors.categories:
	r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
	rgb_color = np.array([r, g, b], dtype=np.uint8)
	category_colors.append(rgb_color)
category_colors.append(np.array([255,255,255], dtype=np.uint8))

def make_image(language, cell_width=16, grid_width=1):
	width = language.shape[0] * cell_width - grid_width
	height = language.shape[0] * cell_width - grid_width
	image = np.full((width,height,3), 255, dtype=np.uint8)
	for (x, y), category in np.ndenumerate(language):
		image[x*cell_width:((x+1)*cell_width)-grid_width,
			  y*cell_width:((y+1)*cell_width)-grid_width] = category_colors[category]
	return image

def save_image(language, output_file, cell_width=16, grid_width=1):
	image = make_image(language, cell_width, grid_width)
	imageio.imsave(output_file, image)

def save_animation(best_chain, animation_path, show_seen=False, create_thumbnail=False):
	generations = best_chain['generations']
	images = []
	if create_thumbnail:
		language = np.array(generations[-1]['language'], dtype=int).reshape((8,8))
		save_image(language, animation_path + '_thumb.gif')
	for gen_i, generation in enumerate(generations):
		language = np.array(generation['language'], dtype=int).reshape((8,8))
		images.append(make_image(language))
		if show_seen:
			training = np.full((8,8), 4, dtype=int)
			for meaning, signal in generation['data_out']:
				training[tuple(meaning)] = signal
			images.append(make_image(training))
	if show_seen:
		imageio.mimsave(animation_path + '.gif', images[:-1], fps=2, loop=1, palettesize=8, subrectangles=True)
	else:
		imageio.mimsave(animation_path + '.gif', images, fps=2, loop=1, palettesize=8, subrectangles=True)
