import matplotlib
matplotlib.use('Agg')

import prettyplotlib as ppl
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1)
data = np.array([0.11917, 0.10789])

ax.set_ylabel('RMSE')
n = 2

ppl.bar(ax, np.arange(n), data, annotate = data.astype('|S7').tolist(), xticklabels = ['1600 clusters', '3000 clusters'])
fig.savefig('clusters.pdf')