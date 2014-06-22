GalaxyZoo
=========

Classifying galaxies using single-layer unsupervised feature extraction.

Pipeline
=========
1. Run kmeans.py on your data (make sure to use modified sklearn/cluster/k_means_.py)
2. Run give_features.m on your training and test sets
3. Run normalize_files(61578, 79975) on your features
4. Run serial_rf.py on the normalized features

Relevant links
==============

http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge

http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf

http://www.stanford.edu/~acoates/