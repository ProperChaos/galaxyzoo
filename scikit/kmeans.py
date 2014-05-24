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
import logging

class KMeansTrainer():
	def __init__(self, images_directory, save_directory, do_rotation_invariant_training, k, patch_width, crop_size, image_size, patches_per_image, compute_whitening, load_whitening, n_iterations = 10, channels = 3):
		assert(compute_whitening ^ load_whitening)

		self.do_rotation_invariant_training = do_rotation_invariant_training
		self.k = k
		self.patch_width = patch_width
		self.patches_per_image = patches_per_image
		self.compute_whitening = compute_whitening
		self.load_whitening = load_whitening
		self.channels = channels
		self.crop_size = crop_size
		self.image_size = image_size
		self.images_directory = images_directory
		self.save_directory = save_directory
		self.n_iterations = n_iterations

		self.logger = logging.getLogger('mainlogger')
		self.logger.setLevel(logging.DEBUG)

		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)

		self.logger.addHandler(ch)

		if self.load_whitening:
			self.M = np.loadtxt(save_directory + '/M.csv', delimiter=",")
			self.P = np.loadtxt(save_directory + '/P.csv', delimiter=",")

	def _rotate_square_vector(self, X, degrees):
		""" Rotates a square vector degrees amount of degrees in steps of 90 degrees. Returns a flattened vector. """

		ret = np.copy(X)
		times = int(round(degrees / 90.0))

		# Loop through patches in X
		for i, o in enumerate(ret):
			temp = np.rot90(o.reshape((self.patch_width, self.patch_width, self.channels), order="F"), times)
			ret[i,:] = np.reshape(temp, self.patch_width**2*self.channels, order="F")

		return ret

	def _yield_random_patch(self, image):
		""" Yields a flattened random patch of size patch_width by patch_width. """

		rand = np.random.random_integers(0, self.image_size-self.patch_width, 2);
		patch = image[rand[0]:rand[0]+self.patch_width, rand[1]:rand[1]+self.patch_width, :]
		vect = np.reshape(patch, self.patch_width**2*self.channels, order="F")
		vectNew = vect - np.mean(vect)

		if sum(vect) != 0:
			vectNew = vectNew / np.sqrt(np.var(vect)+10)

		return vectNew

	def _crop_image(self, image):
		""" Crops the image. Returns a three-dimensional array. """

		startX = (image.shape[0] - self.crop_size) / 2
		startY = (image.shape[0] - self.crop_size) / 2
		endX = startX + self.crop_size
		endY = startY + self.crop_size
		image = image[startX:endX, startY:endY, :]

		return image

	def _resize_image(self, image):
		""" Resizes an image. Returns a three-dimensional array. """
		return scipy.misc.imresize(image, (self.image_size, self.image_size), 'bilinear')

	def whiten(self):
		""" Computes whitening matrices P, Pinv and M. """

		self.logger.debug("Fetching directory listing")

		listing = os.listdir(self.images_directory)
		listing.sort()
		listing = [s for s in listing if s.endswith('.jpg')]

		self.logger.debug("Allocating memory")

		Msum = np.zeros(self.patch_width**2*self.channels)
		Csum = np.zeros((self.patch_width**2*self.channels, self.patch_width**2*self.channels))

		self.logger.debug("Phase 1: Summing individual covariance terms")
		ppi_whitening = self.patches_per_image

		for im in range(0, len(listing)):
			image = misc.imread(self.images_directory + '/' + listing[im])
			image = self._crop_image(image)
			image = self._resize_image(image)
			for patch in range(0, ppi_whitening):
				x = self._yield_random_patch(image)
				Msum += x

				for pixel in range(0, self.patch_width**2*self.channels):
					Csum[pixel, :] += x[pixel] * x

			if im % 100 == 0:
				self.logger.debug("Phase 1 progress: %.2f%%", 100. * im / len(listing))

		M = Msum / len(listing)
		C = np.zeros((self.patch_width**2*self.channels, self.patch_width**2*self.channels))
		
		self.logger.debug("Phase 2: Summing total covariance matrix")

		for i in range(0, self.patch_width**2*self.channels):
			for j in range(0, self.patch_width**2*self.channels):
				C[i, j] = 1./(len(listing) * ppi_whitening) * (Csum[i, j] - M[i]*M[j])

		self.logger.debug("Phase 3: Computing SVD")
		U, S, V = np.linalg.svd(C)

		self.logger.debug("Phase 4: Computing P and Pinv")
		P = np.dot(U, np.dot(np.diag(np.sqrt(1/(S + 0.1))), U.T))
		a = np.diag(np.sqrt(S + 0.1))
		Pinv = np.dot(U, np.dot(a, U.T))

		self.P = P
		self.M = M

		return M, P, Pinv

	def fit(self):
		""" Fits the k-means on the images and returns the centroids. """

		listing = os.listdir(self.images_directory)
		listing.sort()
		listing = [s for s in listing if s.endswith('.jpg')]

		km = MiniBatchKMeans(self.k, init='k-means++', compute_labels=False)
		images_per_batch = math.ceil(self.k / self.patches_per_image)
		X = np.zeros((self.patches_per_image*images_per_batch, self.patch_width**2*self.channels))
		j = 0

		for it in range(0, self.n_iterations):
			for im in range(0, len(listing)):
				image = misc.imread(self.images_directory + '/' + listing[im])
				image = self._crop_image(image)
				image = self._resize_image(image)

				for patch in range(0, self.patches_per_image):
					x = self._yield_random_patch(image)
					X[j*self.patches_per_image+patch, :] = x

				j += 1

				if j < images_per_batch:
					continue

				self.logger.debug("Training k-means: iteration = %i, done = %.2f%%", it, 100. * (im+1) / len(listing))

				j = 0

				if self.do_rotation_invariant_training:
					X90 = self._rotate_square_vector(X, 90)
					X180 = self._rotate_square_vector(X, 180)
					X270 = self._rotate_square_vector(X, 270)

				X = np.dot(X-self.M, self.P) # whitening

				if self.do_rotation_invariant_training:
					X90 = np.dot(X90-self.M, self.P)
					X180 = np.dot(X180-self.M, self.P)
					X270 = np.dot(X270-self.M, self.P)

					km.partial_fit(X, X90, X180, X270)
				else:
					km.partial_fit(X)

		return km.cluster_centers_

	def run_pipeline(self):
		if self.compute_whitening:
			M, P, Pinv = self.whiten()
			np.savetxt(self.save_directory + "/M.csv", M, delimiter=",")
			np.savetxt(self.save_directory + "/P.csv", P, delimiter=",")
			np.savetxt(self.save_directory + "/Pinv.csv", Pinv, delimiter=",")

		centroids = self.fit()

		np.savetxt(self.save_directory + "/centroids.csv", centroids, delimiter=",")

if __name__ == '__main__':
	km = KMeansTrainer('/vagrant/images_training_rev1', '/vagrant/kmeans_results/rot_inv', do_rotation_invariant_training = True, k = 3000, patch_width = 5, crop_size = 150, image_size = 15, patches_per_image = 25, compute_whitening = True, load_whitening = False, n_iterations = 10)
	km.run_pipeline()