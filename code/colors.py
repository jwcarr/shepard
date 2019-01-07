categories = ['#E85A71', '#6B6B7F', '#4EA1D3', '#FCBE32']
categories_lightened = ['#F5B4BF', '#AFAFBC', '#A3CFE8', '#FEE3A8']

# Experimental results
black = '#404040'
light_black = '#DADADA'

# Model simplicity
blue = '#4D5C83'
light_blue = '#DBDEE6'

# Model informativeness
red = '#D65D45'
light_red = '#F5DFDA'


def fake_alpha(hex_color, alpha):
	'''
	Lightens a color, producing a fake alpha transparency effect (for
	the sake of EPS compatibility).
	'''
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
