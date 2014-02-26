import numpy, scipy, sklearn, csv_io, math
from sklearn.ensemble import RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from numpy import array

def RandomForest():
	train = array(csv_io.read_data("features.csv"))
	target = array(csv_io.read_data("solutions.csv"))

	rf = RandomForestRegressor(n_estimators=50, min_samples_split=2, n_jobs=-1)
	scores = cross_validation.cross_val_score(rf, train, target, cv=5, scoring='mean_squared_error');
	scores = -scores;
	rmse = numpy.mean(numpy.sqrt(scores));

	print rmse

def LogisticRegression():
	train = array(csv_io.read_data("features.csv"))
	target = array(csv_io.read_data("solutions.csv"))

	rf = LogisticRegression()
	scores = cross_validation.cross_val_score(rf, train, target, cv=5, scoring='mean_squared_error');
	scores = -scores;
	rmse = numpy.mean(numpy.sqrt(scores));

	print rmse

if __name__ == '__main__':
	LogisticRegression()