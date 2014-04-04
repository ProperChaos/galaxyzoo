import numpy, scipy, sklearn, csv_io, math, csv, itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import linear_model
from multiprocessing import Pool
from numpy import array, loadtxt
import gc

def RandomForest():
	print "Loading features.csv..."
	train = loadtxt('/vol/temp/sreitsma/training_final_ordered_correctly_ffs.csv', delimiter=',')
	gc.collect()
	
	print "Loading solutions.csv..."
	target = loadtxt('solutions.csv', delimiter=',')
	gc.collect()

	train = train[0:25000, :]
	target = target[0:25000, :]

	rf = RandomForestRegressor(n_estimators=50, n_jobs=1)
	rf.fit(train, target)
	joblib.dump(rf, 'classifier.pkl')

def gen_chunks(reader, chunksize=500):
	chunk = []
	for i, line in enumerate(reader):
		if (i % chunksize == 0 and i > 0):
			yield chunk
			del chunk[:]
		chunk.append(line)
	yield chunk

def SGDRegression():
	for a in range(3,4):
		rf = []

		for i in range(0, 37):
			rf.append(sklearn.linear_model.SGDRegressor(loss='huber', epsilon=0.01, penalty = 'l2', alpha = 10**-a, eta0=0.005, shuffle=True,fit_intercept=True))

		for k in range(0,10):
			reader = csv.reader(open('/vol/temp/sreitsma/training.csv', 'rb'))
			reader2 = csv.reader(open('solutions.csv', 'rb'))

			chunkTrain = gen_chunks(reader)
			chunkTarget = gen_chunks(reader2)
			j = 0
			for chunkTr, chunkTa in itertools.izip(chunkTrain, chunkTarget):
				chunkTr = numpy.array(chunkTr)
				chunkTa = numpy.array(chunkTa)

				chunkTr = chunkTr.astype(numpy.float)
				chunkTa = chunkTa.astype(numpy.float)

				print "Processing chunk %i in iteration %i for alpha = %.6f" %(j, k, 10**-a)
				for i in range(0,37):
					rf[i].partial_fit(chunkTr, chunkTa[:, i])
				j += 1

		for i in range(0, 37):
			# Save
			print "Saving classifier %i" %(i)
			numpy.savetxt("model_coef_" + str(a) + "_" + str(i) + ".txt", rf[i].coef_, delimiter=",")
			numpy.savetxt("model_intercept_" + str(a) + "_" + str(i) + ".txt", rf[i].intercept_, delimiter=",")

	print "Done!"
	
def LogisticRegression():
	print "Loading features.csv..."
	train = loadtxt('/vol/temp/sreitsma/training_final_ordered_correctly_ffs.csv', delimiter=',')
	gc.collect()
	
	print "Loading solutions.csv..."
	target_or = loadtxt('solutions.csv', delimiter=',')
	gc.collect()

	for i in range(0, target_or.shape[1]-1):
		train = train[0:1000, :]
		target = target_or[0:1000, i]

		print "Learning classifier #%i" %(i)
		rf = sklearn.linear_model.LogisticRegression()
		rf.fit(train, target)
		
		print "Saving classifier #%i" %(i)
		numpy.savetxt("model_coef_" + str(i) + ".txt", rf[i].coef_, delimiter=",")
		numpy.savetxt("model_intercept_" + str(i) + ".txt", rf[i].intercept_, delimiter=",")
		
def RidgeRegression():
	print "Loading features.csv..."
	train = loadtxt('/vol/temp/sreitsma/training_final_ordered_correctly_ffs.csv', delimiter=',')
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
	SGDRegression()