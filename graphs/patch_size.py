import matplotlib
matplotlib.use('Agg')

import prettyplotlib as ppl
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1)
data = np.array([0.14986, 0.10789, 0.12119])

ax.set_ylabel('RMSE')
n = 3

ppl.bar(ax, np.arange(n), data, annotate = data.astype('|S7').tolist(), xticklabels = ['3x3', '5x5', '10x10'])
fig.savefig('patch_size.pdf')