import numpy, scipy, sklearn, csv_io, math, os
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import OAS
from scipy import misc
from numpy import array
from time import gmtime, strftime, time
from estimate_covariance import get_covariance_estimation

def run_kmeans(image_path, save_path, k, d, w, ppi, M, P):
	now = time()

	listing = os.listdir(image_path)

	km = MiniBatchKMeans(k, init='k-means++', max_iter = 500, batch_size = ppi, compute_labels=False)

	for im in range(0, len(listing)):
		image = misc.imread(image_path + '/' + listing[im])
		X = numpy.zeros((ppi, d*w*w))
		for patch in range(0, ppi):
			x = data_yield(image, w, d)
			X[patch, :] = x

		with open(save_path + "/kmeans.log", "a") as logfile:
			logfile.write('[' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '] #' + str(im) + ' ' + listing[im] + '\n')
		
		#X = X - numpy.asmatrix(X.mean(axis=1)).T
		#X = X / X.std(axis=1)
		#X = numpy.nan_to_num(X)
		#X = numpy.dot(X-M, P)
		km.partial_fit(X);

	numpy.savetxt(save_path + "/centroids.csv", km.cluster_centers_, delimiter=",")
	total = time() - now
	
	with open(save_path + "/kmeans.log", "a") as logfile:
		logfile.write('[' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '] Total time: ' + str(total) + 'seconds\n')
		

def data_yield(im, w, d):
	rand = numpy.random.random_integers(0, 424-w, 2);
	patch = im[rand[0]:rand[0]+w, rand[1]:rand[1]+w, 0:d]
	vect = numpy.reshape(patch, w*w*d, order="F")
	#vect = vect - numpy.mean(vect)

	if sum(vect) != 0:
		vect = vect
		#vect = vect / numpy.sqrt(numpy.var(vect)+10)

	return vect

def build_whitening_matrices(image_path, w, d, ppi):
	listing = os.listdir(image_path)
	listing.sort()
	listing = listing[0:100]

	Msum = numpy.zeros(w*w*d)

	for im in range(0, len(listing)):
		image = misc.imread(image_path + '/' + listing[im])
		X = numpy.zeros((ppi, d*w*w))
		for patch in range(0, ppi):
			x = data_yield(image, w, d)
			x = (x - numpy.mean(x)) / numpy.sqrt(numpy.var(x))
			Msum += x
		print im

	M = Msum / len(listing)

	patches = numpy.zeros((len(listing)*10, d*w*w))
	for im in range(0, len(listing)):
		image = misc.imread(image_path + '/' + listing[im])
		for patch in range(0, 10):
			x = data_yield(image, w, d)
			x = (x - numpy.mean(x)) / numpy.sqrt(numpy.var(x))
			patches[im*10+patch, :] = x
		print im

	patches = patches - numpy.asmatrix(patches.mean(axis=1)).T
	patches = patches / patches.std(axis=1)
	patches = numpy.nan_to_num(patches)
	# Estimate covariance
	oas = OAS()
	oas.fit(patches)
	C = oas.covariance_

	S,V = numpy.linalg.eig(C)
	P = numpy.dot(V, numpy.dot(numpy.diag(numpy.sqrt(1/(S + 0.1))), V.T))

	return M,P

if __name__ == '__main__':
	#M, P = build_whitening_matrices('C:\\Zoo\\images_subset', 12, 3, 10)
	run_kmeans('C:\\Zoo\\images_subset', 'C:\\Zoo\\features', 400, 3, 12, 400, M, P)