from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from math import *
import numpy as np
import logging
import h5py
import csv
import joblib
import cProfile
import time

# Does a RF pass over the data after initial classification. Not necessary but improves results.
class SerialRandomForestModel:
	def __init__(self, n_trees=30, n_jobs=-1, n_sgd_iter=15, chunk_size=100):
		self.sgd_models = []
		self.n_sgd_iter = n_sgd_iter
		self.chunk_size = chunk_size
		self.logger = logging.getLogger('mainlogger')

		for i in range(0, 37):
			self.sgd_models.append(SGDRegressor(loss='huber', penalty='l2', alpha=0.0001, eta0 = 0.005, shuffle=True, fit_intercept=True))

		self.rf_model = RandomForestRegressor(n_estimators = n_trees, n_jobs = n_jobs, max_depth=None, min_samples_split = 2, min_samples_leaf = 1, max_features="auto", bootstrap = True, oob_score=False, verbose = 3)

	def fit(self, training_samples_file, training_solutions_file, train_sgd = True, save=True, save_directory='data_rot_inv/'):
		"""
		Fits the SerialRandomForestModel using the specified data.

		:param training_samples_file: a string filename pointing to a HDF5 file containing a variable named 'f' in which rows are samples and columns are features.
		:param training_solutions_file: a string filename pointing to a .csv file in which rows are samples and columns are predictors.
		:param save: whether to save the models.
		:param save_directory: in which folder to save the models

		"""

		# Train SGD regressor
		if train_sgd:
			self._train_sgd(training_samples_file, training_solutions_file)

			if save:
				self._save_sgd(save_directory, 37) # TODO: not hardcoded

		# Predict using SGD regressor
		self.logger.debug("Predicting using SGD model")
		predictions = self._predict_sgd(training_samples_file, training_solutions_file)
		predictions = predictions.astype(np.float32)

		# Load solutions file
		self.logger.debug("Loading solutions file to memory")
		solutions = np.genfromtxt(training_solutions_file, dtype=float, delimiter=',')
		solutions = solutions.astype(np.float32)

		# Fit random forest
		self.logger.debug("Fitting random forest")
		self.rf_model.fit(predictions, solutions)

		if save:
			self.logger.debug("Random forest model trained successfully, saving to disk")
			joblib.dump(self.rf_model, save_directory + "rf.pkl")

		self.logger.debug("Done")

	def load_sgd_model_from_disk(self, n_predictors, directory='data_rot_var/'):
		for i in range(0, n_predictors):
			self.logger.debug("Loading model %i/%i", i+1, n_predictors)
			self.sgd_models[i].coef_ = np.loadtxt(directory + "model_coef_" + str(i) + ".txt", delimiter=",")
			self.sgd_models[i].intercept_ = np.loadtxt(directory + "model_intercept_" + str(i) + ".txt", delimiter=",")

	def load_rf_model_from_disk(self, directory='data_rot_var/'):
		self.logger.debug("Loading random forest model")
		self.rf_model = joblib.load(directory + "rf.pkl")

	def predict(self, test_samples_file, n_predictors, scale = True, save = True, save_directory = 'data_rot_var/'):
		reader = h5py.File(test_samples_file, 'r')
		result = np.zeros((reader['f'].shape[1], n_predictors))
		j = 0

		for j in range(0, reader['f'].shape[1]):
			sample = reader['f'][:, j].T

			for i in range(0, n_predictors):
				result[j, i] = self.sgd_models[i].predict(sample)
			j += 1

			if j % 100 == 0:
				self.logger.debug("Predicting using SGD model, %.2f%% done", 1.*j/reader['f'].shape[1]*100)

		if save:
			self.logger.debug("Saving SGD result")
			scaled_sgd = np.zeros(result.shape)
			for i in range(0, result.shape[0]):
				sample = result[i, :]
				scaled_sgd[i, :] = self._normalize_sample(sample)
			np.savetxt(save_directory + "sgd_result.csv", scaled_sgd, delimiter=',', fmt='%.6f')

		self.logger.debug("Predicting using random forest model")
		rf_predictions = self.rf_model.predict(result)

		if scale:
			self.logger.debug("Scaling")
			for i in range(0, rf_predictions.shape[0]):
				sample = rf_predictions[i, :]
				result[i, :] = self._normalize_sample(sample)

		if save:
			self.logger.debug("Saving to disk")
			np.savetxt(save_directory + "rf_result.csv", rf_predictions, delimiter=',', fmt='%.6f')

		self.logger.debug("Done")

		return rf_predictions

	def score(self, predictions, solutions):
		self.logger.debug("Computing RMSE and RVSE")
		delta = predictions - solutions
		se = delta**2

		rmse = np.sqrt(np.mean(se))
		rvse = np.sqrt(np.var(se))

		return rmse, rvse

	def _gen_chunks(self, reader):
		chunk = []

		for i, line in enumerate(reader):
			if (i % self.chunk_size == 0 and i > 0):
				yield chunk
				del chunk[:]
			chunk.append(line)
		yield chunk

	def _train_sgd(self, training_samples_file, training_solutions_file):
		for k in range(0, self.n_sgd_iter):
			training_samples_reader = h5py.File(training_samples_file, 'r')
			training_solutions_reader = csv.reader(open(training_solutions_file, 'rb'))

			chunkTarget = self._gen_chunks(training_solutions_reader)
			n_predictors = 0
			n_chunks = ceil(training_samples_reader['f'].shape[1]/self.chunk_size)

			j = 0

			for chunkTa in chunkTarget:
				now = time.time()

				if (j+1)*self.chunk_size > training_samples_reader['f'].shape[1]:
					end_bound = training_samples_reader['f'].shape[1]
				else:
					end_bound = (j+1)*self.chunk_size

				chunkTr = training_samples_reader['f'][:, j*self.chunk_size:end_bound].T
				chunkTa = np.array(chunkTa)
				chunkTa = chunkTa.astype(np.float)

				n_predictors = chunkTa.shape[1]

				for i in range(0, n_predictors):
					self.sgd_models[i].partial_fit(chunkTr, chunkTa[:, i])

				self.logger.debug("Processed chunk %i/%i in iteration %i/%i, took %.2f seconds", j+1, n_chunks, k+1, self.n_sgd_iter, time.time() - now)

				j += 1

		self.logger.debug("SGD model trained successfully")

	def _save_sgd(self, save_directory, n_predictors):
		for i in range(0, n_predictors):
			# Save
			self.logger.debug("Saving SGD model for predictor %i/%i", i+1, n_predictors)

			np.savetxt(save_directory + "model_coef_" + str(i) + ".txt", self.sgd_models[i].coef_, delimiter=',')
			np.savetxt(save_directory + "model_intercept_" + str(i) + ".txt", self.sgd_models[i].intercept_, delimiter=',')

	def _get_n_predictors(self, training_solutions_file):
		training_solutions_reader = csv.reader(open(training_solutions_file, 'rb'))
		chunkTarget = self._gen_chunks(training_solutions_reader)
		chunkTa = chunkTarget.next()
		chunkTa = np.array(chunkTa)
		chunkTa = chunkTa.astype(np.float)
		return chunkTa.shape[1]

	def _get_n_chunks(self, training_samples_file):
		training_samples_reader = h5py.File(training_samples_file, 'r')
		return ceil(training_samples_reader['f'].shape[1]/self.chunk_size)

	def _predict_sgd(self, training_samples_file, training_solutions_file):
		n_predictors = self._get_n_predictors(training_solutions_file)
		n_chunks = self._get_n_chunks(training_samples_file)

		training_samples_reader = h5py.File(training_samples_file, 'r')
		n_samples = training_samples_reader['f'].shape[1]

		predictions = np.zeros((n_samples, n_predictors))

		for j in range(0, int(n_chunks)+1):
			if (j+1)*self.chunk_size > training_samples_reader['f'].shape[1]:
				end_bound = training_samples_reader['f'].shape[1]
			else:
				end_bound = (j+1)*self.chunk_size

			chunkTr = training_samples_reader['f'][:, j*self.chunk_size:end_bound].T

			self.logger.debug("Predicting chunk %i/%i", j, n_chunks)

			for i in range(0, n_predictors):
				predictions[j*self.chunk_size:end_bound, i] = self.sgd_models[i].predict(chunkTr)
			
		return predictions

	def _normalize_sample(self, sample):
		# Regularization epsilon
		epsilon = 10**-8

		# Clip to range
		sample[sample < 0] = 0
		sample[sample > 1] = 1

		# Normalize Class 1
		total = sample[0] + sample[1] + sample[2]
		factor = 1 / (total + epsilon)

		sample[0] *= factor
		sample[1] *= factor
		sample[2] *= factor

		# Normalize Class 2
		total = (sample[3] + sample[4]) / (sample[1] + epsilon)
		factor = 1 / (total + epsilon)

		sample[3] *= factor
		sample[4] *= factor

		# Normalize Class 3
		total = (sample[5] + sample[6]) / (sample[4] + epsilon)
		factor = 1 / (total + epsilon)

		sample[5] *= factor
		sample[6] *= factor

		# Normalize Class 4
		total = (sample[7] + sample[8]) / (sample[4] + epsilon)
		factor = 1 / (total + epsilon)

		sample[7] *= factor
		sample[8] *= factor

		# Normalize Class 5
		total = (sample[9] + sample[10] + sample[11] + sample[12]) / (sample[4] + epsilon)
		factor = 1 / (total + epsilon)

		sample[9] *= factor
		sample[10] *= factor
		sample[11] *= factor
		sample[12] *= factor

		# Normalize Class 6
		total = sample[13] + sample[14]
		factor = 1 / (total + epsilon)

		sample[13] *= factor
		sample[14] *= factor

		# Normalize Class 7
		total = (sample[15] + sample[16] + sample[17]) / (sample[0] + epsilon)
		factor = 1 / (total + epsilon)

		sample[15] *= factor
		sample[16] *= factor
		sample[17] *= factor

		# Normalize Class 8
		total = (sample[18] + sample[19] + sample[20] + sample[21] + sample[22] + sample[23] + sample[24]) / (sample[13] + epsilon)
		factor = 1 / (total + epsilon)

		sample[18] *= factor
		sample[19] *= factor
		sample[20] *= factor
		sample[21] *= factor
		sample[22] *= factor
		sample[23] *= factor
		sample[24] *= factor

		# Normalize Class 9
		total = (sample[25] + sample[26] + sample[27]) / (sample[3] + epsilon)
		factor = 1 / (total + epsilon)

		sample[25] *= factor
		sample[26] *= factor
		sample[27] *= factor

		# Normalize Class 10
		total = (sample[28] + sample[29] + sample[30]) / (sample[7] + epsilon)
		factor = 1 / (total + epsilon)

		sample[28] *= factor
		sample[29] *= factor
		sample[30] *= factor

		# Normalize Class 11
		total = (sample[31] + sample[32] + sample[33] + sample[34] + sample[35] + sample[36]) / (sample[7] + epsilon)
		factor = 1 / (total + epsilon)

		sample[31] *= factor
		sample[32] *= factor
		sample[33] *= factor
		sample[34] *= factor
		sample[35] *= factor
		sample[36] *= factor

		return np.nan_to_num(sample)

if __name__ == '__main__':
	srfm = SerialRandomForestModel(n_jobs = 1)

	# Debug + fit
	logger = logging.getLogger('mainlogger')
	logger.setLevel(logging.DEBUG)

	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)

	logger.addHandler(ch)

	srfm.load_sgd_model_from_disk(n_predictors = 37)
	srfm.load_rf_model_from_disk()

	#srfm.fit('../rot_inv/TRAINING_norma.mat', 'solutions.csv', train_sgd = True)
	predictions = srfm.predict('../rot_var/TRAINING_norma.mat', n_predictors = 37)

	solutions = np.genfromtxt('solutions.csv', dtype=float, delimiter=',')
	solutions = solutions.astype(np.float32)
	solutions = solutions[0:5000, :]
	predictions = predictions[0:5000, :]

	delta = predictions - solutions
	se = delta**2

	np.savetxt("error_var.csv", se, delimiter=',', fmt='%.6f')

	rmse, rvse = srfm.score(predictions, solutions)

	logger.debug('RMSE: %.6f', rmse)
	logger.debug('RVSE: %.6f', rvse)