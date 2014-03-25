import numpy, scipy, sklearn, csv_io, math, csv, itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn import linear_model
from multiprocessing import Pool
from numpy import array

def RandomForest():
	train = array(csv_io.read_data("features_whitened.csv"))
	target = array(csv_io.read_data("solutions.csv"))
	target = target[0:99, :]

	rf = RandomForestRegressor(n_estimators=50, min_samples_split=2, n_jobs=-1)
	scores = cross_validation.cross_val_score(rf, train, target, cv=5, scoring='mean_squared_error')
	scores = -scores
	rmse = numpy.mean(numpy.sqrt(scores))

	print rmse

def gen_chunks(reader, chunksize=500):
	chunk = []
	for i, line in enumerate(reader):
		if (i % chunksize == 0 and i > 0):
			yield chunk
			del chunk[:]
		chunk.append(line)
	yield chunk

def SGDRegression():
	print "Loading features.csv..."
	#train = array(csv_io.read_data("C:\\Zoo\\Code\\galaxyzoo\\scikit\\features.csv"))
	reader = csv.reader(open('features.csv', 'rb'))

	print "Loading solutions.csv..."
	#target_or = array(csv_io.read_data("solutions.csv"))
	reader2 = csv.reader(open('solutions.csv', 'rb'))

	total = 0
	rf = []

	for i in range(0, 37):
		rf.append(sklearn.linear_model.SGDRegressor(eta0=0.001, n_iter=100))

	j = 0
	chunkTrain = gen_chunks(reader)
	chunkTarget = gen_chunks(reader2)
	for chunkTr, chunkTa in itertools.izip(chunkTrain, chunkTarget):
		chunkTr = numpy.array(chunkTr)
		chunkTa = numpy.array(chunkTa)

		chunkTr = chunkTr.astype(numpy.float)
		chunkTa = chunkTa.astype(numpy.float)

		print "Processing chunk %i" %(j)
		for i in range(0,37):
			rf[i].partial_fit(chunkTr, chunkTa[:, i])
		j += 1

	reader = csv.reader(open('features.csv', 'rb'))
	reader2 = csv.reader(open('solutions.csv', 'rb'))

	test = gen_chunks(reader)
	test_target = gen_chunks(reader2)
	chunk = next(test)
	chunk_target = next(test_target)
	chunk = numpy.array(chunk).astype(numpy.float)
	chunk_target = numpy.array(chunk_target).astype(numpy.float)

	for i in range(0, 37):
		predict = rf[i].predict(chunk)
		delta = (predict-chunk_target[:,i])**2
		rmse = numpy.sum(delta)

		total += rmse

		# Save
		print "Saving classifier %i" %(i)
		numpy.savetxt("model_coef_" + str(i) + ".txt", rf[i].coef_, delimiter=",")
		numpy.savetxt("model_intercept_" + str(i) + ".txt", rf[i].intercept_, delimiter=",")

	total = numpy.sqrt(total / 37 / 500)

	print "Done! RMSE: %.5f" %(total)

if __name__ == '__main__':
	SGDRegression()