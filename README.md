# Quadric Inclusion Programming Demo

This demo regenerates the result figure from ThomasSentis2019TAC.

## Getting Started

The instructions here are based on a successful CodeOcean setup environment (with the same name: ThomasSentis2019TAC).

### Working CodeOcean Environment

Using Python 3.6.3 in Ubuntu 16.04 through an Anaconda python environment.
We will need to complile some things, so
```
apt-get build-essential cmake g++ gcc gfortran libopenblas-dev
```
And some things from conda will work (but decidedly not all of the dependencies can be safely obtained through conda).
Using python-control, cyclus, and cvxgrp as additional conda channels:
```
conda install lapack=3.5.0 matplotlib=2.0.2 nose=1.3.7 numpy=1.15.4 python=3.6.6 scipy=1.2.0
```
And some post-installation is necessary to get the rest
```
pip install --upgrade pip
pip install scikit-build

git clone https://github.com/python-control/Slycot.git slycot
cd slycot
python setup.py install 
# I forgot to cd .. (shouldn't matter)

pip install control

pip install cvxopt cvxpy

git clone https://github.com/GrayThomas/control_LMIs.git
cd control_LMIs
python setup.py install
ls
cd ..
```

This ought to allow main.py to run, which will generate image files in ../results/ (or throw an error message if that directory does not exist.)


## Authors

* **Gray Thomas** - *Initial work* - [Gray Thomas](https://github.com/graythomas)

See also the list of [contributors](https://github.com/GrayThomas/ThomasSentis2019TAC/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* cvxpy
* python-control