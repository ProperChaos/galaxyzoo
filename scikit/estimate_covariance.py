import numpy
from sklearn.covariance import OAS

def get_covariance_estimation(sample_matrix):
	oas = OAS()
	oas.fit(sample_matrix)
	return oas.covariance_