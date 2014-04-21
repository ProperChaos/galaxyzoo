#!/bin/sh

apt-get update
apt-get install pip python-dev build-essential libatlas-dev libatlas-base-dev liblapack3gf liblapack-dev gfortran git libjpeg-dev
ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
ln -s /usr/lib/x86_64-linux-gnu/libz.so /usr/lib
pip install numpy
pip install pil
pip install scipy
git clone git://github.com/amueller/scikit-learn.git
cd scikit-learn
git checkout remotes/origin/minibatch_reallocation_fixes #of zoiets
export PYTHONPATH=$PYTHONPATH:/home/vagrant/scikit-learn
python setup.py build_ext --inplace
