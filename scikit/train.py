import numpy, scipy, sklearn, csv_io, math
from sklearn.ensemble import RandomForestRegressor

def main():
	train = csv_io.read_data("features.csv")
	target = csv_io.read_data("solutions.csv")
	test = csv_io.read_data("features_test.csv")
	test_target = csv_io.read_data("solutions_test.csv")

	rf = RandomForestRegressor(n_estimators=50, min_samples_split=2, n_jobs=-1)

	rf.fit(train, target)

	predicted = rf.predict(test)
	delta = test_target - predicted
	delta = delta ** 2

	rmse = math.sqrt(numpy.mean(delta))
	print rmse

if __name__ == '__main__':
	main()