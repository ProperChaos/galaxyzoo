import numpy, scipy, sklearn, csv_io, math, os
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import OAS
from scipy import misc
from numpy import array
from time import gmtime, strftime, time
from estimate_covariance import get_covariance_estimation

def run_kmeans(image_path, save_path, k, d, w, ppi, M, P, Pinv):
	now = time()

	listing = os.listdir(image_path)
	listing.sort()

	km = MiniBatchKMeans(k, init='k-means++', max_iter = 500, batch_size = ppi, compute_labels=False)

	for im in range(0, len(listing)):
		image = misc.imread(image_path + '/' + listing[im])
		X = numpy.zeros((ppi, d*w*w))
		for patch in range(0, ppi):
			x = data_yield(image, w, d)
			X[patch, :] = x

		#with open(save_path + "/kmeans.log", "a") as logfile:
		#	logfile.write('[' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '] #' + str(im) + ' ' + listing[im] + '\n')
		
		# Local normalization (per patch)
		X = X - numpy.asmatrix(X.mean(axis=1)).T
		X = X / numpy.sqrt(X.var(axis=1) + 10)
		X = numpy.nan_to_num(X)
		#X = numpy.dot(X-M, P) # whitening
		km.partial_fit(X)

		if im % 100 == 0:
			print "Training k-means: %.2f%%" %(100. * im / len(listing))

	centroids = km.cluster_centers_

	numpy.savetxt(save_path + "/centroids.csv", centroids, delimiter=",")
	numpy.savetxt(save_path + "/whitening_P.csv", P, delimiter=",")
	numpy.savetxt(save_path + "/whitening_Pinv.csv", Pinv, delimiter=",")
	numpy.savetxt(save_path + "/whitening_M.csv", M, delimiter=",")
	total = time() - now
	
	with open(save_path + "/kmeans.log", "a") as logfile:
		logfile.write('[' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '] Total time: ' + str(total) + 'seconds\n')
		

def data_yield(im, w, d):
	rand = numpy.random.random_integers(0, 424-w, 2);
	patch = im[rand[0]:rand[0]+w, rand[1]:rand[1]+w, 0:d]
	vect = numpy.reshape(patch, w*w*d, order="F")
	vect = vect - numpy.mean(vect)

	if sum(vect) != 0:
		vect = vect / numpy.sqrt(numpy.var(vect)+10)

	return vect

def build_whitening_matrices(image_path, w, d, ppi):
	listing = os.listdir(image_path)
	listing.sort()

	Msum = numpy.zeros(w*w*d)
	Csum = numpy.zeros((w*w*d, w*w*d))

	for im in range(0, len(listing)):
		image = misc.imread(image_path + '/' + listing[im])
		X = numpy.zeros((ppi, d*w*w))
		for patch in range(0, ppi/10):
			x = data_yield(image, w, d)
			Msum += x
			for pixel in range(0, w*w*d):
				Csum[pixel, :] += x[pixel] * x
		if im % 100 == 0:
			print "Building whitening matrices (M, C): %.2f%%" %(100. * im / len(listing))

	M = Msum / len(listing)
	C = numpy.zeros((w*w*d, w*w*d))
	
	print "Summing covariance matrix C..."

	for i in range(0, w*w*d):
		for j in range(0, w*w*d):
			C[i, j] = 1./(len(listing) * ppi/10) * (Csum[i, j] - M[i]*M[j])

	# Msum = numpy.zeros(w*w*d)

	# for im in range(0, len(listing)):
	# 	image = misc.imread(image_path + '/' + listing[im])
	# 	X = numpy.zeros((ppi, d*w*w))
	# 	for patch in range(0, ppi):
	# 		x = data_yield(image, w, d)
	# 		Msum += x
	# 	print im

	# M = Msum / len(listing)

	# patches = numpy.zeros((len(listing)*(len(listing)/10), d*w*w))
	# for im in range(0, len(listing)):
	# 	image = misc.imread(image_path + '/' + listing[im])
	# 	for patch in range(0, ppi/10):
	# 		x = data_yield(image, w, d)
	# 		patches[im*10+patch, :] = x
	# 	print im

	# patches = numpy.nan_to_num(patches)
	# # Estimate covariance
	# oas = OAS()
	# oas.fit(patches)
	# C = oas.covariance_

	print "Computing eigenvalues and eigenvectors of C..."
	S,V = numpy.linalg.eig(C)

	print "Computing P..."
	P = numpy.dot(V, numpy.dot(numpy.diag(numpy.sqrt(1/(S + 0.1))), V.T))

	print "Computing Pinv..."
	a = numpy.diag(1/numpy.sqrt(1/(S + 0.1)))
	Pinv = numpy.dot(V, numpy.dot(a, V.T))

	return M,P,Pinv

if __name__ == '__main__':
	subset = 'C:\\Zoo\\images_subset'
	complete = 'C:\\Zoo\\images_training_rev1'

	#M, P, Pinv = build_whitening_matrices(complete, 12, 3, 10)
	M = P = Pinv = None
	run_kmeans(complete, 'C:\\Zoo\\features', 400, 3, 12, 400, M, P, Pinv)