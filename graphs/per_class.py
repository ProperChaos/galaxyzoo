import matplotlib
matplotlib.use('Agg')

import prettyplotlib as ppl
import matplotlib.pyplot as plt
import numpy as np

data1 = np.genfromtxt('error_inv.csv', dtype=float, delimiter=',')
data2 = np.genfromtxt('error_var.csv', dtype=float, delimiter=',')

for i in range(0, data1.shape[0]):
	if np.sum(data1[i, 0:2]) != 0:
		data1[i, 0:2] = data1[i, 0:2] / np.sum(data1[i, 0:2])
	if np.sum(data1[i, 3:4]) != 0:
		data1[i, 3:4] = data1[i, 3:4] / np.sum(data1[i, 3:4])
	if np.sum(data1[i, 5:6]) != 0:
		data1[i, 5:6] = data1[i, 5:6] / np.sum(data1[i, 5:6])
	if np.sum(data1[i, 7:8]) != 0:
		data1[i, 7:8] = data1[i, 7:8] / np.sum(data1[i, 7:8])
	if np.sum(data1[i, 9:12]) != 0:
		data1[i, 9:12] = data1[i, 9:12] / np.sum(data1[i, 9:12])
	if np.sum(data1[i, 13:14]) != 0:
		data1[i, 13:14] = data1[i, 13:14] / np.sum(data1[i, 13:14])
	if np.sum(data1[i, 15:17]) != 0:
		data1[i, 15:17] = data1[i, 15:17] / np.sum(data1[i, 15:17])
	if np.sum(data1[i, 18:24]) != 0:
		data1[i, 18:24] = data1[i, 18:24] / np.sum(data1[i, 18:24])
	if np.sum(data1[i, 25:27]) != 0:
		data1[i, 25:27] = data1[i, 25:27] / np.sum(data1[i, 25:27])
	if np.sum(data1[i, 28:30]) != 0:
		data1[i, 28:30] = data1[i, 28:30] / np.sum(data1[i, 28:30])
	if np.sum(data1[i, 31:36]) != 0:
		data1[i, 31:36] = data1[i, 31:36] / np.sum(data1[i, 31:36])

for i in range(0, data2.shape[0]):
	if np.sum(data2[i, 0:2]) != 0:
		data2[i, 0:2] = data2[i, 0:2] / np.sum(data2[i, 0:2])
	if np.sum(data2[i, 3:4]) != 0:
		data2[i, 3:4] = data2[i, 3:4] / np.sum(data2[i, 3:4])
	if np.sum(data2[i, 5:6]) != 0:
		data2[i, 5:6] = data2[i, 5:6] / np.sum(data2[i, 5:6])
	if np.sum(data2[i, 7:8]) != 0:
		data2[i, 7:8] = data2[i, 7:8] / np.sum(data2[i, 7:8])
	if np.sum(data2[i, 9:12]) != 0:
		data2[i, 9:12] = data2[i, 9:12] / np.sum(data2[i, 9:12])
	if np.sum(data2[i, 13:14]) != 0:
		data2[i, 13:14] = data2[i, 13:14] / np.sum(data2[i, 13:14])
	if np.sum(data2[i, 15:17]) != 0:
		data2[i, 15:17] = data2[i, 15:17] / np.sum(data2[i, 15:17])
	if np.sum(data2[i, 18:24]) != 0:
		data2[i, 18:24] = data2[i, 18:24] / np.sum(data2[i, 18:24])
	if np.sum(data2[i, 25:27]) != 0:
		data2[i, 25:27] = data2[i, 25:27] / np.sum(data2[i, 25:27])
	if np.sum(data2[i, 28:30]) != 0:
		data2[i, 28:30] = data2[i, 28:30] / np.sum(data2[i, 28:30])
	if np.sum(data2[i, 31:36]) != 0:
		data2[i, 31:36] = data2[i, 31:36] / np.sum(data2[i, 31:36])

delta_q1 = np.sqrt(np.mean(data1[:, 0:2])) - np.sqrt(np.mean(data2[:, 0:2]))
delta_q2 = np.sqrt(np.mean(data1[:, 3:4])) - np.sqrt(np.mean(data2[:, 3:4]))
delta_q3 = np.sqrt(np.mean(data1[:, 5:6])) - np.sqrt(np.mean(data2[:, 5:6]))
delta_q4 = np.sqrt(np.mean(data1[:, 7:8])) - np.sqrt(np.mean(data2[:, 7:8]))
delta_q5 = np.sqrt(np.mean(data1[:, 9:12])) - np.sqrt(np.mean(data2[:, 9:12]))
delta_q6 = np.sqrt(np.mean(data1[:, 13:14])) - np.sqrt(np.mean(data2[:, 13:14]))
delta_q7 = np.sqrt(np.mean(data1[:, 15:17])) - np.sqrt(np.mean(data2[:, 15:17]))
delta_q8 = np.sqrt(np.mean(data1[:, 18:24])) - np.sqrt(np.mean(data2[:, 18:24]))
delta_q9 = np.sqrt(np.mean(data1[:, 25:27])) - np.sqrt(np.mean(data2[:, 25:27]))
delta_q10 = np.sqrt(np.mean(data1[:, 28:30])) - np.sqrt(np.mean(data2[:, 28:30]))
delta_q11 = np.sqrt(np.mean(data1[:, 31:36])) - np.sqrt(np.mean(data2[:, 31:36]))

delta = [delta_q1, delta_q2, delta_q3, delta_q4, delta_q5, delta_q6, delta_q7, delta_q8, delta_q9, delta_q10, delta_q11]

delta = np.array(delta) - 0.00038

fig, ax = plt.subplots(1)

ax.set_xlabel('Decision tree question')
ax.set_ylabel('Difference in root mean squared error')
ax.set_xlim(0, 11)
ax.get_xaxis().set_ticks([])
ax.yaxis.set_ticks(np.arange(-0.0015, 0.0008, 0.0001))

ppl.bar(ax, np.arange(11), delta, annotate = (np.arange(11)+1).tolist())

ax.yaxis.set_ticks_position('left')
fig.savefig('per_class.pdf')