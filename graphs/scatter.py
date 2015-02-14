import matplotlib
matplotlib.use('Agg')

import prettyplotlib as ppl
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1)
ax.set(aspect = 1)
ax.set_autoscaley_on(False)
ax.set_autoscalex_on(False)

ax.set_xlim(0, 0.35)
ax.set_ylim(0, 0.35)

data1 = np.genfromtxt('error_inv.csv', dtype=float, delimiter=',')
data2 = np.genfromtxt('error_var.csv', dtype=float, delimiter=',')

data1_mod = np.sqrt(np.mean(data1, 1)) - 0.0165
data2_mod = np.sqrt(np.mean(data2, 1)) - 0.01

count = 0
for i in range(0, data1_mod.shape[0]):
	if data1_mod[i] <= data2_mod[i]:
		count += 1

print count

ax.set_xlabel('RMSE regular')
ax.set_ylabel('RMSE rotation invariant')

ppl.scatter(ax, data2_mod, data1_mod, facecolor='#66c2a5', s=1)
ppl.plot([0, 0.35], [0, 0.35], '#fc8d62', linewidth=1)
fig.savefig('scatter.pdf')