import numpy, scipy, sklearn, csv_io, math, csv, itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import linear_model
from multiprocessing import Pool
from numpy import array, loadtxt
import gc

def normalize_sample(sample):
	# Set negative to 0, for some reason numpy.clip does not work
	for c in range(0, sample.shape[0]):
		if sample[c] < 0:
			sample[c] = 0

	# Normalize Class 1
	total = sample[0] + sample[1] + sample[2]
	factor = 1 / total

	sample[0] *= factor
	sample[1] *= factor
	sample[2] *= factor

	# Normalize Class 2
	total = (sample[3] + sample[4]) / sample[1]
	factor = 1 / total

	sample[3] *= factor
	sample[4] *= factor

	# Normalize Class 3
	total = (sample[5] + sample[6]) / sample[4]
	factor = 1 / total

	sample[5] *= factor
	sample[6] *= factor

	# Normalize Class 4
	total = (sample[7] + sample[8]) / sample[4]
	factor = 1 / total

	sample[7] *= factor
	sample[8] *= factor

	# Normalize Class 5
	total = (sample[9] + sample[10] + sample[11] + sample[12]) / sample[4]
	factor = 1 / total

	sample[9] *= factor
	sample[10] *= factor
	sample[11] *= factor
	sample[12] *= factor

	# Normalize Class 6
	total = sample[13] + sample[14]
	factor = 1 / total

	sample[13] *= factor
	sample[14] *= factor

	# Normalize Class 7
	total = (sample[15] + sample[16] + sample[17]) / sample[0]
	factor = 1 / total

	sample[15] *= factor
	sample[16] *= factor
	sample[17] *= factor

	# Normalize Class 8
	total = (sample[18] + sample[19] + sample[20] + sample[21] + sample[22] + sample[23] + sample[24]) / sample[13]
	factor = 1 / total

	sample[18] *= factor
	sample[19] *= factor
	sample[20] *= factor
	sample[21] *= factor
	sample[22] *= factor
	sample[23] *= factor
	sample[24] *= factor

	# Normalize Class 9
	total = (sample[25] + sample[26] + sample[27]) / sample[3]
	factor = 1 / total

	sample[25] *= factor
	sample[26] *= factor
	sample[27] *= factor

	# Normalize Class 10
	total = (sample[28] + sample[29] + sample[30]) / sample[7]
	factor = 1 / total

	sample[28] *= factor
	sample[29] *= factor
	sample[30] *= factor

	# Normalize Class 11
	total = (sample[31] + sample[32] + sample[33] + sample[34] + sample[35] + sample[36]) / sample[7]
	factor = 1 / total

	sample[31] *= factor
	sample[32] *= factor
	sample[33] *= factor
	sample[34] *= factor
	sample[35] *= factor
	sample[36] *= factor

	return numpy.nan_to_num(sample)

if __name__ == '__main__':
	print "Scaling..."
	result = loadtxt('C:\Zoo\Code\galaxyzoo\scikit\\result_no_scaling_3.csv', delimiter=',')
	for i in range(0, result.shape[0]):
		sample = result[i, :]
		sample = normalize_sample(sample)

		if i % 100 == 0:
			print 100. * i / result.shape[0]

	numpy.savetxt("result_3.csv", result, delimiter=',', fmt='%.8f')