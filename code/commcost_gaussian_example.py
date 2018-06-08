import matplotlib.pyplot as plt
from matplotlib import gridspec
import commcost
import colors

gammas = [0.01, 1, 'inf']
metric = 'circle_euclidean'
y_max = 0.07

N = commcost.Need((64,))
P = commcost.Partition([0]*16 + [1]*16 + [2]*16 + [3]*16)
S = P.spawn_speaker()

fig = plt.figure(figsize=(5.5, 2))
grid = gridspec.GridSpec(nrows=1, ncols=len(gammas))
for i in range(len(gammas)):
	sp = fig.add_subplot(grid[0, i])
	sp.set_axisbelow(True)

	if gammas[i] == 'inf':
		L = P.spawn_listener(999999999)
		K = commcost.cost(P, N, S, L)
		plt.title('$\\gamma = \\infty, K = ' + str(int(K)) + '$', fontsize=10)
	else:
		L = P.spawn_listener(gammas[i], metric)
		K = commcost.cost(P, N, S, L)
		plt.title('$\\gamma = ' + str(gammas[i]) + ', K = ' + str(round(K, 3)) + '$', fontsize=10)
	for j in range(len(P)):
		category_distribution = list(L[j].probabilities.flatten())
		category_distribution = [category_distribution[-1]] + category_distribution + [category_distribution[0]]
		sp.plot(range(0, P.size+2), category_distribution, linewidth=2.0, color=colors.categories[j])
	plt.xticks(fontsize=7)
	plt.yticks(fontsize=7)
	plt.tick_params(axis='x', which='both', bottom='off', top='off')
	plt.tick_params(axis='y', which='both', left='off', right='off')
	if i == 0:
		plt.ylabel('$C_j(i)$', fontsize=10)
	else:
		sp.set_yticklabels([])
	if i == len(gammas)//2:
		plt.xlabel('$i\\in\\mathcal{U}$', fontsize=10)
	plt.xlim(0, P.size+1)
	y_max = y_max * 1.05
	y_min = 0 - (y_max * 0.05)
	plt.ylim(y_min, y_max)
	ticks = [1] + [(P.size//len(P))*j for j in range(1, len(P)+1)]
	sp.set_xticks(ticks)
grid.tight_layout(fig, pad=0.1, h_pad=0.1, w_pad=0.5)
plt.savefig('../visuals/gaussians.svg')
fig.clf()
