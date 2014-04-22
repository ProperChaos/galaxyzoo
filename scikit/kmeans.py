import numpy as np
import scipy, sklearn, csv_io, math, os, time
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import OAS
from scipy import misc
from numpy import array
from estimate_covariance import get_covariance_estimation
import shutil
from joblib import Parallel, delayed
import tempfile
import joblib

def run_kmeans(image_path, save_path, k, d, w, ppi, ipb, M, P, Pinv):
	listing = os.listdir(image_path)
	listing.sort()
	listing = [s for s in listing if s.endswith('.jpg')]

	km = MiniBatchKMeans(k, init='k-means++', max_iter = 100, batch_size = ppi, compute_labels=False)
	X = np.zeros((ppi*ipb, d*w*w))
	j = 0

	for it in range(0, 10):
		for im in range(0, len(listing)):
			image = misc.imread(image_path + '/' + listing[im])
			image = crop_image(image)

			for patch in range(0, ppi):
				x = data_yield(image, w, d)
				X[j*ppi+patch, :] = x

			j += 1

			if j < ipb:
				continue

			print "Training k-means: iteration = %i, done = %.2f%%" %(it, 100. * im / len(listing))

			j = 0
			X = np.dot(X-M, P) # whitening

			km.partial_fit(X)

	centroids = km.cluster_centers_

	np.savetxt(save_path + "/centroids.csv", centroids, delimiter=",")

def run_spherical_kmeans(image_path, save_path, k, d, w, ppi, ipb, M, P, Pinv):
	listing = os.listdir(image_path)
	listing.sort()
	listing = [s for s in listing if s.endswith('.jpg')]

	X = np.zeros((ppi*len(listing), d*w*w))

	print "Filling matrix..."

	for im in range(0, len(listing)):
		image = misc.imread(image_path + '/' + listing[im])
		image = crop_image(image)

		for patch in range(0, ppi):
			x = data_yield(image, w, d)
			X[im*ppi+patch, :] = x

		if im % 100 == 0:
			print im

	print "Whitening..."
	X = np.dot(X-M, P)

	print "Start training spherical kmeans..."

	centroids = spherical_kmeans(X, k, 20, 5000)

	np.savetxt(save_path + "/centroids.csv", centroids, delimiter=',')

def spherical_kmeans(X, k, n_iter, batch_size=1000):
    """
    Do a spherical k-means.  Line by line port of Coates' matlab code.

    Returns a (k, n_pixels) centroids matrix
    """

    # shape (n_samples, 1)
    x2 = np.sum(X**2, 1, keepdims=True)

    # randomly initialize centroids
    #centroids = np.random.randn(k, X.shape[1]) * 0.1
    centroids = X[1:k, :]

    for iteration in range(1, n_iter + 1):
        # shape (k, 1)
        c2 = 0.5 * np.sum(centroids ** 2, 1, keepdims=True)

        # shape (k, n_pixels)
        summation = np.zeros((k, X.shape[1]))
        counts = np.zeros((k, 1))
        loss = 0

        for i in range(0, X.shape[0], batch_size):
            last_index = min(i + batch_size, X.shape[0])
            m = last_index - i

            # shape (k, batch_size) - shape (k, 1)
            tmp = np.abs(np.dot(centroids, X[i:last_index, :].T) - c2)
            # shape (batch_size, )
            indices = np.argmax(tmp, 0)
            distances = tmp[indices, range(batch_size)]
            sims = np.argsort(distances)

            # shape (1, batch_size)
            val = np.max(tmp, 0, keepdims=True)

            loss += np.sum((0.5 * x2[i:last_index]) - val.T)

            # Don't use a sparse matrix here
            S = np.zeros((batch_size, k))
            S[range(batch_size), indices] = 1

            # shape (k, n_pixels)
            this_sum = np.dot(S.T, X[i:last_index, :])
            summation += this_sum

            this_counts = np.sum(S, 0, keepdims=True).T
            counts += this_counts

        # Sometimes raises RuntimeWarnings because some counts can be 0
        centroids = summation / counts

        bad_indices = np.where(counts <= 1)[0]
        print bad_indices.shape

        e = 0
        for r in range(0, bad_indices.shape[0]):
        	centroids[bad_indices[r], :] = X[sims[e], :]
        	e += 1

        assert not np.any(np.isnan(centroids))

        print "K-means iteration {} of {}, loss {}".format(iteration, n_iter, loss)
    return centroids

def data_yield(im, w, d):
	rand = np.random.random_integers(0, 15-w, 2);
	patch = im[rand[0]:rand[0]+w, rand[1]:rand[1]+w, 0:d]
	vect = np.reshape(patch, w*w*d, order="F")
	vectNew = vect - np.mean(vect)

	if sum(vect) != 0:
		vectNew = vectNew / np.sqrt(np.var(vect)+10)

	return vectNew

def crop_image(image):
	# Crop here
	crop_size = 150
	startX = (image.shape[0] - crop_size) / 2
	startY = (image.shape[0] - crop_size) / 2
	endX = startX + crop_size
	endY = startY + crop_size
	image = image[startX:endX, startY:endY, :]

	image = scipy.misc.imresize(image, (15, 15), 'bilinear')

	return image

def build_whitening_matrices(image_path, save_path, w, d, ppi):
	print "Retrieving directory listing..."
	listing = os.listdir(image_path)
	listing.sort()
	listing = [s for s in listing if s.endswith('.jpg')]

	print "Pre-allocating memory..."
	Msum = np.zeros(w*w*d)
	Csum = np.zeros((w*w*d, w*w*d))

	print "Start building whitening matrices..."
	for im in range(0, len(listing)):
		image = misc.imread(image_path + '/' + listing[im])
		image = crop_image(image)
		for patch in range(0, ppi):
			x = data_yield(image, w, d)
			Msum += x
			for pixel in range(0, w*w*d):
				Csum[pixel, :] += x[pixel] * x
		if im % 1 == 0:
			print listing[im]
			print "Building whitening matrices (M, C): %.2f%%" %(100. * im / len(listing))

	M = Msum / len(listing)
	C = np.zeros((w*w*d, w*w*d))
	
	print "Summing covariance matrix C..."

	for i in range(0, w*w*d):
		for j in range(0, w*w*d):
			C[i, j] = 1./(len(listing) * ppi) * (Csum[i, j] - M[i]*M[j])

	# Msum = np.zeros(w*w*d)

	# for im in range(0, len(listing)):
	# 	image = misc.imread(image_path + '/' + listing[im])
	# 	X = np.zeros((ppi, d*w*w))
	# 	for patch in range(0, ppi):
	# 		x = data_yield(image, w, d)
	# 		Msum += x
	# 	print im

	# M = Msum / len(listing)

	# patches = np.zeros((len(listing)*(len(listing)/10), d*w*w))
	# for im in range(0, len(listing)):
	# 	image = misc.imread(image_path + '/' + listing[im])
	# 	for patch in range(0, ppi/10):
	# 		x = data_yield(image, w, d)
	# 		patches[im*10+patch, :] = x
	# 	print im

	# patches = np.nan_to_num(patches)
	# # Estimate covariance
	# oas = OAS()
	# oas.fit(patches)
	# C = oas.covariance_

	print "Computing eigenvalues and eigenvectors of C..."
	U,S,V = np.linalg.svd(C)

	print "Computing P..."
	P = np.dot(U, np.dot(np.diag(np.sqrt(1/(S + 0.1))), U.T))

	print "Computing Pinv..."
	a = np.diag(np.sqrt(S + 0.1))
	Pinv = np.dot(U, np.dot(a, U.T))

	np.savetxt(save_path + "/whitening_P.csv", P, delimiter=",")
	np.savetxt(save_path + "/whitening_Pinv.csv", Pinv, delimiter=",")
	np.savetxt(save_path + "/whitening_M.csv", M, delimiter=",")

	return M,P,Pinv

if __name__ == '__main__':
	root = '/home/vagrant/data'
	subset = root + '/images_subset'
	complete = '/vagrant/images_training_rev1'
	save_path = root + '/features';
	w = 5
	d = 3
	k = 3000
	ppi_learning = 10
	ipb = 120
	ppi_whitening = 20

	#M, P, Pinv = build_whitening_matrices(complete, save_path, w, d, ppi_whitening)
	run_spherical_kmeans(subset, save_path, k, d, w, ppi_learning, ipb, np.loadtxt(save_path + '/whitening_M.csv', delimiter=","), np.loadtxt(save_path + '/whitening_P.csv', delimiter=","), np.loadtxt(save_path + '/whitening_Pinv.csv', delimiter=","))