import numpy, scipy, sklearn, csv_io, math, os
from sklearn.cluster import MiniBatchKMeans
from scipy import misc
from numpy import array
from time import gmtime, strftime, time

def main():
	now = time()

	image_path = 'C:/Zoo/images_subset' #'/vol/temp/sreitsma/images'
	save_path = 'C:/Zoo/features'
	k = 400
	d = 3
	w = 12
	ppi = 400

	listing = os.listdir(image_path)

	km = MiniBatchKMeans(k, init='k-means++', max_iter = 1000, batch_size = ppi, compute_labels=False)

	for im in range(0, len(listing)):
		image = misc.imread(image_path + '/' + listing[im])
		X = numpy.zeros((ppi, d*w*w))
		for patch in range(0, ppi):
			x = data_yield(image, w, d)
			X[patch, :] = x

		with open(save_path + "/kmeans.log", "a") as logfile:
			logfile.write('[' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '] #' + str(im) + ' ' + listing[im] + '\n')
		
		km.partial_fit(X);

	numpy.savetxt("C:/Zoo/features/centroids_newww.csv", km.cluster_centers_, delimiter=",")
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

if __name__ == '__main__':
	main()