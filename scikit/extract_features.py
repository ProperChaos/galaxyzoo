import numpy, scipy, sklearn, csv_io, math, os, multiprocessing
from multiprocessing import Value, Lock, Pool
from scipy import misc
from skimage import util
from numpy import array
from time import gmtime, strftime, time, sleep

# Because Python can't pickle class methods?
def unpack_wrapper(x):
	return calc_feature(*x)

# Extracts feature vector from an image, given centroids
def calc_feature(centroids, patch_width, stride, path, p, q):
	t = time()
	image = misc.imread(path)

	# Crop here
	crop_size = 300
	startX = (image.shape[0] - crop_size) / 2
	startY = (image.shape[0] - crop_size) / 2
	endX = startX + crop_size
	endY = startY + crop_size
	image = image[startX:endX, startY:endY, :]

	# Extract patches
	patches = patch_extract(image, patch_width, stride)
	patches = numpy.float32(patches)

	# Preprocessing
	# Normalize
	patches = patches - numpy.asmatrix(patches.mean(axis=1)).T
	patches = patches / patches.std(axis=1)
	patches = numpy.nan_to_num(patches)

	# Triangle (soft) activation function
	xx = numpy.sum(numpy.exp2(patches), axis=1)
	cc = numpy.sum(numpy.exp2(centroids), axis=1)
	xc = 2*numpy.dot(patches, numpy.transpose(centroids))

	z = numpy.sqrt(cc + (xx - xc))
	mu = z.mean(axis=1)
	patches = numpy.maximum(0, mu-z)

	# Reshape to 2D plane before pooling
	rows = image.shape[0] - patch_width + 1
	cols = image.shape[1] - patch_width + 1
	patches = numpy.array(patches, copy=False).reshape(rows, cols, centroids.shape[0], order="F")

	# Pool
	half_rows = round(rows / 2)
	half_cols = round(cols / 2)

	# Calculate pool values
	q1 = numpy.sum(numpy.sum(patches[1:half_rows, 1:half_cols, :], 0), 0)
	q2 = numpy.sum(numpy.sum(patches[half_rows+1:patches.shape[0], 1:half_cols, :], 0), 0)
	q3 = numpy.sum(numpy.sum(patches[1:half_rows, half_cols+1:patches.shape[1], :], 0), 0)
	q4 = numpy.sum(numpy.sum(patches[half_rows+1:patches.shape[0], half_cols+1:patches.shape[1], :], 0), 0)

	# Print time
	#print "Finished %s, took %.2f seconds" %(path, time() - t)

	output = numpy.transpose(numpy.append(q1, numpy.append(q2, numpy.append(q3, q4))))

	# Put output in queue (so that it is sent to the original thread)
	q.put((p, output))

	# Concatenate and return
	return 0

def patch_extract(image, patch_width, stride):
	# Ugly code, extracts r, g, b patches from the image convolutionally.

	patches_r = util.view_as_windows(image[:, :, 0], (patch_width, patch_width), stride)
	patches_g = util.view_as_windows(image[:, :, 1], (patch_width, patch_width), stride)
	patches_b = util.view_as_windows(image[:, :, 2], (patch_width, patch_width), stride)

	patches_r = numpy.reshape(patches_r, ((image.shape[0]-patch_width+1)**2, patch_width, patch_width), order="F")
	patches_r = numpy.reshape(patches_r, ((image.shape[0]-patch_width+1)**2, patch_width*patch_width), order="F")

	patches_g = numpy.reshape(patches_g, ((image.shape[0]-patch_width+1)**2, patch_width, patch_width), order="F")
	patches_g = numpy.reshape(patches_g, ((image.shape[0]-patch_width+1)**2, patch_width*patch_width), order="F")

	patches_b = numpy.reshape(patches_b, ((image.shape[0]-patch_width+1)**2, patch_width, patch_width), order="F")
	patches_b = numpy.reshape(patches_b, ((image.shape[0]-patch_width+1)**2, patch_width*patch_width), order="F")

	patches = numpy.append(patches_r, patches_g, 1);
	patches = numpy.append(patches, patches_b, 1);

	return patches

def give_features(image_path, centroids, patch_width, stride):
	# Get directory contents
	listing = os.listdir(image_path)
	listing.sort()

	# Reserve memory
	features = numpy.zeros((len(listing), 4*centroids.shape[0]))

	# Multiprocessing
	pool = Pool(12)
	to = 1000 #len(listing)
	q = multiprocessing.Manager().Queue()
	pool_input = [(centroids, patch_width, stride, image_path + '/' + listing[p], p, q) for p in range(0, to)]

	t = time()

	output = pool.map_async(unpack_wrapper, pool_input)
	last = 0

	while not output.ready():
		if q.qsize() != last:
			print "%.2f%% finished, ETA: %.2f minutes, mean time per image: %.2f seconds" %(float(q.qsize()) / to * 100, float(time() - t) / q.qsize() * (to-q.qsize()) / 60, float(time() - t) / q.qsize())
			last = q.qsize()
		sleep(0.5)

	# Ready!
	# Normalize
	writeToFile(q)

def writeToFile(q):
	l = []
	while q.qsize() > 0:
		g = q.get()
		l.append(g)

	l.sort(key=lambda s: s[0])
	k=l
	k = [item[1] for item in l]

	arr = numpy.array(k)
	arr = arr - numpy.transpose(numpy.asmatrix(arr.mean(axis=1)))
	arr = arr / numpy.asmatrix(arr.std(axis=1))

	numpy.savetxt("/vol/temp/sreitsma/features_new.csv", arr, delimiter=",")

def main():
	centroids = array(csv_io.read_data("centroids.csv"))
	give_features('/vol/temp/sreitsma/images_training_rev1', centroids, 12, 1)

if __name__ == '__main__':
	main()