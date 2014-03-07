import numpy, scipy, sklearn, csv_io, math
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn import linear_model
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

def LogisticRegression():
	train = array(csv_io.read_data("features_whitened.csv"))
	target_or = array(csv_io.read_data("solutions.csv"))
	total = 0

	for i in range(0, target_or.shape[1]-1):
		target = target_or[0:99, i]

		rf = sklearn.linear_model.LogisticRegression()
		rf.fit(train, target)
		scores = rf.predict(train)
		rmse = numpy.sqrt(numpy.mean((scores-target)**2))
		print "MAX: %.5f | CURRENT: %.5f | GOODNESS: %.5f %%" %(numpy.sqrt(numpy.mean((target)**2)), rmse, (1-(rmse / numpy.sqrt(numpy.mean((target)**2)))) * 100)

		#scores = cross_validation.cross_val_score(rf, train, target, cv=5, scoring='mean_squared_error');
		#scores = -scores;
		#rmse = numpy.mean(numpy.sqrt(scores));

		total += rmse

	print total / (target_or.shape[1]-1)

if __name__ == '__main__':
	RandomForest()