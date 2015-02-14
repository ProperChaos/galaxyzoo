GalaxyZoo
=========

Classifying galaxies using single-layer unsupervised feature extraction.

Pipeline
=========
1. Run kmeans.py on your data (make sure to use modified sklearn/cluster/k_means_.py for rotation invariance)
2. Run give_features.m on your training and test sets (or run extract_features.py, both should work)
3. Standardize the features (divide by covariance, subtract mean)
4. Run serial_rf.py on the normalized features

Relevant files
=========
* scikit/kmeans.py (extracts random patches, clusters using k-means)
* scikit/extract_features.py (extracts features from images and computes distance to centroids, multithreaded)
* scikit/serial_rf.py (runs an SGD regressor and a random forest regressor, only the SGD is strictly needed)

Note that lots of these scripts could have been much cleaner (and shorter) if the data would have been preprocessed into a more usable format or if pandas was used.

Relevant links
==============

http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge

http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf

http://www.stanford.edu/~acoates/