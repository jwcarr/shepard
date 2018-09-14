'''
Code for visualizing IL chain trajectories in simplicity-
informativeness space. This was not included in the paper; refer to my
thesis instead.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.misc import comb
import tools
import commcost
import rectlang

plt.rcParams['svg.fonttype'] = 'none'

def transform_paper3_data():
	r_space = rectlang.Space((6,8))
	c_space = commcost.Space((6,8))
	old_new = {0:(1,3), 1:(0,2), 2:(0,6), 3:(2,5), 4:(5,1), 5:(3,7), 6:(0,5), 7:(5,7), 8:(4,2), 9:(5,2), 10:(4,0), 11:(0,0), 12:(5,6), 13:(4,3), 14:(1,4), 15:(4,4), 16:(1,1), 17:(4,1), 18:(2,4), 19:(5,4), 20:(4,5), 21:(3,0), 22:(5,5), 23:(3,3), 24:(2,6), 25:(3,5), 26:(0,4), 27:(0,7), 28:(2,1), 29:(3,1), 30:(3,4), 31:(3,2), 32:(4,6), 33:(1,2), 34:(0,1), 35:(3,6), 36:(1,7), 37:(1,0), 38:(4,7), 39:(2,2), 40:(2,7), 41:(2,0), 42:(1,5), 43:(1,6), 44:(5,0), 45:(5,3), 46:(2,3), 47:(0,3)}
	data = {'chains':[]}
	for chain in ['I', 'J', 'K', 'L']:
		chain_data = {'chain_id':chain, 'generations':[]}
		for generation in range(11):
			lang = np.full((6,8), -1, dtype=int)
			words = {}
			with open('/Users/jon/Code/flatlanders/data/experiment_3/%s/%is'%(chain, generation)) as file:
				for i, line in enumerate(file):
					word = line.split('\t')[0]
					if word in words:
						words[word].append(i)
					else:
						words[word] = [i]
			for cat, (key, values) in enumerate(words.items(), 0):
				for value in values:
					coord = old_new[value]
					lang[coord] = cat
			comp = r_space.complexity(lang)
			cost = c_space.cost(lang)
			chain_data['generations'].append({'prod_cost':cost, 'prod_complexity':comp})
		data['chains'].append(chain_data)
	print(data)

def bernstein_poly(i, n, t):
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=128):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return np.column_stack((xvals, yvals))

def draw_color_curve(axis, points):
	curve = bezier_curve(points)
	curve = curve.reshape(-1, 1, 2)
	segments = np.hstack([curve[:-1], curve[1:]])
	coll = LineCollection(segments, cmap=plt.get_cmap('viridis_r'))
	coll.set_array(np.arange(curve.shape[0]))
	axis.add_collection(coll)
	axis.scatter([points[0][0]], [points[0][1]], color='#440256')
	axis.scatter([points[-1][0]], [points[-1][1]], color='#FDE725')

def simulate(axis, shape, nsims, maxcats, xlim, ylim):
	c_space = commcost.Space(shape)
	r_space = rectlang.Space(shape)
	for n_cats in range(2, maxcats+1):
		color = str(((maxcats+2) - (n_cats-1)) / (maxcats+2))
		X, Y = [], []
		for i in range(nsims):
			_, part = commcost.random_partition(shape, n_cats, True)
			comp = r_space.complexity(part)
			cost = c_space.cost(part)
			if comp < xlim[1] and cost > ylim[0]:
				X.append(comp)
				Y.append(cost)
			_, part = commcost.random_partition(shape, n_cats, False)
			comp = r_space.complexity(part)
			cost = c_space.cost(part)
			if comp < xlim[1] and cost > ylim[0]:
				X.append(comp)
				Y.append(cost)
		axis.scatter(X, Y, color=color)
	return X, Y

def plot_space(axis, il_data, shape, xlim, ylim, nsims=1, maxcats=8):
	sim_x, sim_y = simulate(axis, shape, nsims, maxcats, xlim, ylim)
	for chain in il_data['chains']:
		points = []
		for g, generation in enumerate(chain['generations']):
			if g == chain['first_fixation']:
				break
			points.append((generation['prod_complexity'], generation['prod_cost']))
		draw_color_curve(axis, points)
	axis.set_xlim(*xlim)
	axis.set_ylim(*ylim)
	axis.set_xlabel('Simplicity ⬌ Complexity', )
	axis.set_ylabel('Informativeness ⬌ Communicative cost')

def plot(datasets, shape, nsims, maxcats, figure_path, figsize=(5,4.8)):
	fig, axes = plt.subplots(len(datasets), 1, figsize=figsize, squeeze=False)
	for (dataset, xlim, ylim), axis in zip(datasets, axes.flatten()):
		plot_space(axis, dataset, shape, xlim, ylim, nsims, maxcats)
	fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5)
	fig.savefig(figure_path, format='svg')
	tools.format_svg_labels(figure_path)
	if not figure_path.endswith('.svg'):
		tools.convert_svg(figure_path, figure_path)


if __name__ == '__main__':

	paper2_experiment_2 = tools.read_json_file('../data/experiments/exp2_chains.json')
	plot([(paper2_experiment_2, (0,600), (4.25,6.25))], (8,8), 40, 8, '/Users/jon/Desktop/simp_inf_space_paper2.eps', (5, 3.5))

	paper3_experiment_1 = {'chains': [{'chain_id':'A', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.8013146362322552, 'prod_complexity': 500.3523506476383}, {'prod_cost': 2.3135097896464076, 'prod_complexity': 469.7393706318256}, {'prod_cost': 3.5761337253216983, 'prod_complexity': 420.4555350509122}, {'prod_cost': 3.7664607957008585, 'prod_complexity': 430.3123021670949}, {'prod_cost': 3.885832620809545, 'prod_complexity': 367.8183070485775}, {'prod_cost': 3.821246055428005, 'prod_complexity': 360.27225368655166}, {'prod_cost': 3.7010641740232333, 'prod_complexity': 335.0118234632523}, {'prod_cost': 3.8544227015928465, 'prod_complexity': 271.15845956361323}, {'prod_cost': 3.9194886848495143, 'prod_complexity': 346.5993599076374}, {'prod_cost': 3.8236870317184244, 'prod_complexity': 272.28129174107806}]}, {'chain_id':'B', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.7775941354199247, 'prod_complexity': 500.3523506476383}, {'prod_cost': 3.015339516082009, 'prod_complexity': 383.590513042486}, {'prod_cost': 3.2038578054268956, 'prod_complexity': 415.70806474884756}, {'prod_cost': 4.057812571738381, 'prod_complexity': 333.90665862026856}, {'prod_cost': 3.8368710599481024, 'prod_complexity': 312.68183632261565}, {'prod_cost': 4.194237549414243, 'prod_complexity': 249.30711910430784}, {'prod_cost': 3.994860671376986, 'prod_complexity': 234.9880425888588}, {'prod_cost': 4.073143030197608, 'prod_complexity': 171.9270068776173}, {'prod_cost': 4.183324251545299, 'prod_complexity': 257.01229057538103}, {'prod_cost': 4.116609634562112, 'prod_complexity': 246.21700216066344}]}, {'chain_id':'C', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.9228988385651775, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.6612048112732047, 'prod_complexity': 500.3523506476383}, {'prod_cost': 2.0582754751812375, 'prod_complexity': 459.8826035156429}, {'prod_cost': 3.0376949297620843, 'prod_complexity': 480.63881641527297}, {'prod_cost': 2.803407728623185, 'prod_complexity': 427.5388541716279}, {'prod_cost': 2.6576556063639796, 'prod_complexity': 479.5961377480083}, {'prod_cost': 4.1213562038828275, 'prod_complexity': 311.06508838610915}, {'prod_cost': 4.1143242935542474, 'prod_complexity': 267.5338214390134}, {'prod_cost': 4.398334511761342, 'prod_complexity': 275.8860168646472}, {'prod_cost': 4.017827549227469, 'prod_complexity': 280.8066963766235}]}, {'chain_id':'D', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.5001663560017602, 'prod_complexity': 500.3523506476383}, {'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 2.6856309325441634, 'prod_complexity': 480.6388164152729}, {'prod_cost': 4.369534874265101, 'prod_complexity': 355.1880919369278}, {'prod_cost': 4.943244224855613, 'prod_complexity': 162.86466323819914}, {'prod_cost': 5.264238895911901, 'prod_complexity': 123.25086973658702}, {'prod_cost': 5.251073110430511, 'prod_complexity': 169.22718125662323}, {'prod_cost': 5.351810303750186, 'prod_complexity': 100.98637899166779}, {'prod_cost': 5.60162120054561, 'prod_complexity': 5.044394119358453}, {'prod_cost': 5.60162120054561, 'prod_complexity': 5.044394119358453}]}]}
	paper3_experiment_2 = {'chains': [{'chain_id':'E', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.5976849718899833, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.9934862185323943, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.9143655954109362, 'prod_complexity': 480.6388164152729}, {'prod_cost': 2.3316930337897395, 'prod_complexity': 480.6388164152729}, {'prod_cost': 2.4598154126083003, 'prod_complexity': 490.4955835314556}, {'prod_cost': 2.99023024906897, 'prod_complexity': 460.92528218290755}, {'prod_cost': 2.7383782412926543, 'prod_complexity': 437.3956212878106}, {'prod_cost': 1.932405299012936, 'prod_complexity': 480.6388164152729}, {'prod_cost': 3.1678261425808536, 'prod_complexity': 451.0685150667249}, {'prod_cost': 2.5836321308694767, 'prod_complexity': 490.4955835314556}]}, {'chain_id':'F', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.554753758146888, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.7027863629336615, 'prod_complexity': 510.2091177638209}, {'prod_cost': 2.1248855169725505, 'prod_complexity': 469.7393706318256}, {'prod_cost': 2.7385805214106558, 'prod_complexity': 490.49558353145557}, {'prod_cost': 2.3041393591108283, 'prod_complexity': 500.3523506476383}, {'prod_cost': 1.9506288832089222, 'prod_complexity': 480.6388164152729}, {'prod_cost': 2.0832988891383075, 'prod_complexity': 510.2091177638209}, {'prod_cost': 2.191718802582226, 'prod_complexity': 490.4955835314556}, {'prod_cost': 2.90808847511179, 'prod_complexity': 460.92528218290755}, {'prod_cost': 2.844478079837864, 'prod_complexity': 458.83992484837825}]}, {'chain_id':'G', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.5955955015223506, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.9027032948377842, 'prod_complexity': 490.49558353145557}, {'prod_cost': 2.2033729221793985, 'prod_complexity': 468.0086013036233}, {'prod_cost': 2.509492949887763, 'prod_complexity': 490.49558353145557}, {'prod_cost': 2.734836890554979, 'prod_complexity': 459.8826035156429}, {'prod_cost': 2.8655072564788604, 'prod_complexity': 396.92587415581517}, {'prod_cost': 2.7271051749095214, 'prod_complexity': 470.78204929909026}, {'prod_cost': 3.1619061932273964, 'prod_complexity': 369.02385595975045}, {'prod_cost': 3.0515580477325823, 'prod_complexity': 448.9206715564764}, {'prod_cost': 2.850047513991274, 'prod_complexity': 430.3123021670949}]}, {'chain_id':'H', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.5198126477971883, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.5426871402016507, 'prod_complexity': 500.3523506476383}, {'prod_cost': 1.8159928520501507, 'prod_complexity': 469.7393706318256}, {'prod_cost': 2.730209584651898, 'prod_complexity': 459.8826035156429}, {'prod_cost': 2.541044913886293, 'prod_complexity': 450.0258363994602}, {'prod_cost': 2.587359142421134, 'prod_complexity': 428.1644586568465}, {'prod_cost': 2.7131673235110325, 'prod_complexity': 458.83992484837825}, {'prod_cost': 2.369480597704553, 'prod_complexity': 415.7080647488475}, {'prod_cost': 2.3961902936957165, 'prod_complexity': 448.92067155647646}, {'prod_cost': 2.0192391379625083, 'prod_complexity': 500.3523506476383}]}]}
	paper3_experiment_3 = {'chains': [{'chain_id':'I', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 2.1579748592722545, 'prod_complexity': 448.9831577321956}, {'prod_cost': 2.380719523613929, 'prod_complexity': 490.49558353145557}, {'prod_cost': 2.4769840487440042, 'prod_complexity': 470.7820492990902}, {'prod_cost': 3.7554860637569427, 'prod_complexity': 377.67507416476013}, {'prod_cost': 3.7591442523358998, 'prod_complexity': 312.57050129468394}, {'prod_cost': 3.9460717156637894, 'prod_complexity': 339.5160407869216}, {'prod_cost': 3.9809071543303824, 'prod_complexity': 335.6374279484708}, {'prod_cost': 3.877125447890335, 'prod_complexity': 326.46875149322574}, {'prod_cost': 3.405607576866643, 'prod_complexity': 331.9951224893899}, {'prod_cost': 3.179300768587799, 'prod_complexity': 398.59415730829846}]}, {'chain_id':'J', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.5204811565959886, 'prod_complexity': 510.2091177638209}, {'prod_cost': 2.104041979533773, 'prod_complexity': 490.4955835314556}, {'prod_cost': 1.764259678915584, 'prod_complexity': 470.78204929909026}, {'prod_cost': 2.042479471170782, 'prod_complexity': 490.49558353145557}, {'prod_cost': 2.766531245679683, 'prod_complexity': 429.26962349983023}, {'prod_cost': 1.6440306933562159, 'prod_complexity': 490.49558353145557}, {'prod_cost': 2.2867693300844114, 'prod_complexity': 459.88260351564287}, {'prod_cost': 2.1636636986266575, 'prod_complexity': 459.8826035156429}, {'prod_cost': 2.203165488943009, 'prod_complexity': 459.8826035156429}, {'prod_cost': 1.967786454032142, 'prod_complexity': 500.3523506476382}]}, {'chain_id':'K', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 2.472917502855669, 'prod_complexity': 490.49558353145557}, {'prod_cost': 1.7756146467339402, 'prod_complexity': 490.49558353145557}, {'prod_cost': 3.6842378715920274, 'prod_complexity': 378.2550185907146}, {'prod_cost': 2.9924636972803222, 'prod_complexity': 376.63239549749534}, {'prod_cost': 3.2833430940274972, 'prod_complexity': 408.86799860652724}, {'prod_cost': 3.63216909584813, 'prod_complexity': 344.45151639738884}, {'prod_cost': 3.6443752089353145, 'prod_complexity': 320.6853021356421}, {'prod_cost': 3.204437055400505, 'prod_complexity': 369.02385595975045}, {'prod_cost': 3.4021851260899707, 'prod_complexity': 367.29308663154814}, {'prod_cost': 3.299136846330258, 'prod_complexity': 387.06910703963257}]}, {'chain_id':'L', 'first_fixation':None, 'generations': [{'prod_cost': 1.436063501088083, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.5125616800582145, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.5111293243615602, 'prod_complexity': 510.2091177638209}, {'prod_cost': 1.658609784635324, 'prod_complexity': 490.49558353145557}, {'prod_cost': 1.8880274781663688, 'prod_complexity': 479.59613774800823}, {'prod_cost': 2.0452193507813226, 'prod_complexity': 480.6388164152729}, {'prod_cost': 2.0386241011430117, 'prod_complexity': 480.63881641527286}, {'prod_cost': 3.5435700760628457, 'prod_complexity': 398.59415730829846}, {'prod_cost': 3.282063203594441, 'prod_complexity': 258.35417726103293}, {'prod_cost': 4.057895192856267, 'prod_complexity': 294.1250021492108}, {'prod_cost': 3.4699180621603163, 'prod_complexity': 303.98261048341936}]}]}

	plot([(paper3_experiment_1, (0,520), (1,6)),
	      (paper3_experiment_3, (0,520), (1,6))], (6,8), 5, 48, '/Users/jon/Desktop/simp_inf_space_paper3.eps', (5, 7))
