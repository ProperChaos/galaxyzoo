import matplotlib
matplotlib.use('Agg')

import prettyplotlib as ppl
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1)
data = np.array([0.11495, 0.10789])

ax.set_ylabel('RMSE')
n = 2

ppl.bar(ax, np.arange(n), data, annotate = data.astype('|S7').tolist(), xticklabels = ['No whitening', 'Whitening'])
fig.savefig('whitening.pdf')