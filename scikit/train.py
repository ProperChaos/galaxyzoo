import numpy, scipy, sklearn, csv_io, math, csv, itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn import linear_model
from multiprocessing import Pool
from numpy import array, loadtxt
import gc

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
	reader = csv.reader(open('/vol/temp/sreitsma/training_final.csv', 'rb'))

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

	for i in range(0, 37):
		# Save
		print "Saving classifier %i" %(i)
		numpy.savetxt("model_coef_" + str(i) + ".txt", rf[i].coef_, delimiter=",")
		numpy.savetxt("model_intercept_" + str(i) + ".txt", rf[i].intercept_, delimiter=",")

	print "Done!"
	
def LogisticRegression():
	print "Loading features.csv..."
	train = loadtxt('/vol/temp/sreitsma/training_final.csv', delimiter=',')
	gc.collect()
	
	print "Loading solutions.csv..."
	target_or = loadtxt('solutions.csv', delimiter=',')
	gc.collect()

	for i in range(0, target_or.shape[1]-1):
		target = target_or[0:61500, i]

		print "Learning classifier #%i" %(i)
		rf = sklearn.linear_model.LogisticRegression()
		rf.fit(train, target)
		
		print "Saving classifier #%i" %(i)
		numpy.savetxt("model_coef_" + str(i) + ".txt", rf[i].coef_, delimiter=",")
		numpy.savetxt("model_intercept_" + str(i) + ".txt", rf[i].intercept_, delimiter=",")
		
def RidgeRegression():
	print "Loading features.csv..."
	train = loadtxt('/vol/temp/sreitsma/training_final.csv', delimiter=',')
	gc.collect()
	
	print "Loading solutions.csv..."
	target_or = loadtxt('solutions.csv', delimiter=',')
	gc.collect()

	target = target_or[0:61500, :]

	print "Learning classifier"
	rf = sklearn.linear_model.Ridge(copy_X = False)
	rf.fit(train, target)
	
	print "Saving classifier"
	numpy.savetxt("model_coef" + ".txt", rf.coef_, delimiter=",")

if __name__ == '__main__':
	RidgeRegression()