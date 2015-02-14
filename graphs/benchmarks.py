import matplotlib
matplotlib.use('Agg')

import prettyplotlib as ppl
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1)
data = np.array([0.27160, 0.16194, 0.10256, 0.07492])

ax.set_ylabel('RMSE')
n = 4

ppl.bar(ax, np.arange(n), data, annotate = data.astype('|S7').tolist(),
	xticklabels = ['All zeros\nbenchmark', 'Central\npixel benchmark', 'Thesis score', 'Kaggle winner'],
	color = [ppl.colors.set2[0], ppl.colors.set2[0], ppl.colors.set2[1], ppl.colors.set2[0]])
fig.savefig('benchmarks.pdf')