import matplotlib
matplotlib.use('Agg')

import prettyplotlib as ppl
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1)
data = np.array([0.10789, 0.09923])

ax.set_ylabel('RMSE')
n = 2

ppl.bar(ax, np.arange(n), data, annotate = data.astype('|S7').tolist(), xticklabels = ['Regular', 'Rotation invariant'])
fig.savefig('rotation_invariance.pdf')