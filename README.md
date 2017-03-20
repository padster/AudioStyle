# AudioStyle
UBC 540 Project: Style transfer for Audio

Install Lasagne & everything needed
(http://lasagne.readthedocs.io/en/latest/user/installation.html)

pip install numpy
pip install scipy
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

pip install audioread

Or read this: http://deeplearning.net/software/theano/install.html

If using BLAS for multithreading:
1) sudo apt-get install libatlas-base-dev
2) OMP_NUM_THREADS=6 python run.py
