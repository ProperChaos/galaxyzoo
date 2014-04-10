import numpy, scipy, sklearn, csv_io, math, os, time
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import OAS
from scipy import misc
from numpy import array
from estimate_covariance import get_covariance_estimation

def run_kmeans(image_path, save_path, k, d, w, ppi, M, P, Pinv):
	listing = os.listdir(image_path)
	listing.sort()
	listing = [s for s in listing if s.endswith('.jpg')]

	km = MiniBatchKMeans(k, init='random', max_iter = 1, batch_size = ppi, compute_labels=False)

	for im in range(0, len(listing)):
		image = misc.imread(image_path + '/' + listing[im])
		image = crop_image(image)
		X = numpy.zeros((ppi, d*w*w))
		for patch in range(0, ppi):
			x = data_yield(image, w, d)
			X[patch, :] = x

		X = numpy.dot(X-M, P) # whitening

		km.partial_fit(X)

		if im % 1 == 0:
			print "Training k-means: %.2f%%" %(100. * im / len(listing))

			plt.figure()
			for i in range(0, 10):
				plt.subplot(5, 5, i+1)
				cc = numpy.dot(X, Pinv) + M
				p = numpy.reshape(cc[i, :], (10, 10, 3))
				p = (p - numpy.min(p)) / (numpy.max(p) - numpy.min(p))
				plt.imshow(p, interpolation='nearest')
				plt.axis('off')

			clusters = numpy.dot(km.cluster_centers_, Pinv) + M
			print numpy.mean(clusters)
			for i in range(10, 20):
				plt.subplot(5, 5, i+1)

				p = numpy.reshape(clusters[i-10, :], (10, 10, 3))
				p = (p - numpy.min(p)) / (numpy.max(p) - numpy.min(p))
				plt.imshow(p, interpolation='nearest')
				plt.axis('off')

			plt.show()

	centroids = km.cluster_centers_

	numpy.savetxt(save_path + "/centroids.csv", centroids, delimiter=",")

def data_yield(im, w, d):
	rand = numpy.random.random_integers(0, 207-w, 2);
	patch = im[rand[0]:rand[0]+w, rand[1]:rand[1]+w, 0:d]
	vect = numpy.reshape(patch, w*w*d)
	vectNew = vect - numpy.mean(vect)

	if sum(vect) != 0:
		vectNew = vectNew / numpy.sqrt(numpy.var(vect)+10)

	return vectNew

def crop_image(image):
	# Crop here
	crop_size = 207
	startX = (image.shape[0] - crop_size) / 2
	startY = (image.shape[0] - crop_size) / 2
	endX = startX + crop_size
	endY = startY + crop_size
	image = image[startX:endX, startY:endY, :]

	#image = scipy.misc.imresize(image, (69, 69), 'bilinear')

	return image

def build_whitening_matrices(image_path, save_path, w, d, ppi):
	listing = os.listdir(image_path)
	listing.sort()
	listing = [s for s in listing if s.endswith('.jpg')]

	Msum = numpy.zeros(w*w*d)
	Csum = numpy.zeros((w*w*d, w*w*d))

	for im in range(0, len(listing)):
		image = misc.imread(image_path + '/' + listing[im])
		image = crop_image(image)
		for patch in range(0, ppi):
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
			C[i, j] = 1./(len(listing) * ppi) * (Csum[i, j] - M[i]*M[j])

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
	U,S,V = numpy.linalg.svd(C)

	print "Computing P..."
	P = numpy.dot(U, numpy.dot(numpy.diag(numpy.sqrt(1/(S + 0.1))), U.T))

	print "Computing Pinv..."
	a = numpy.diag(numpy.sqrt(S + 0.1))
	Pinv = numpy.dot(U, numpy.dot(a, U.T))

	numpy.savetxt(save_path + "/whitening_P.csv", P, delimiter=",")
	numpy.savetxt(save_path + "/whitening_Pinv.csv", Pinv, delimiter=",")
	numpy.savetxt(save_path + "/whitening_M.csv", M, delimiter=",")

	return M,P,Pinv

if __name__ == '__main__':
	subset = 'C:\\Zoo\\images_subset'
	complete = 'C:\\Zoo\\images_training_rev1'
	save_path = 'C:\\Zoo\\features\\new';

	#M, P, Pinv = build_whitening_matrices(subset, save_path, 10, 3, 10)
	run_kmeans(subset, save_path, 10, 3, 10, 10, numpy.loadtxt(save_path + '/whitening_M.csv', delimiter=","), numpy.loadtxt(save_path + '/whitening_P.csv', delimiter=","), numpy.loadtxt(save_path + '/whitening_Pinv.csv', delimiter=","))