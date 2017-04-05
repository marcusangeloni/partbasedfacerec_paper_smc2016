# Part-Based Representation and Classification for Face Recognition

This repository contains the source code of face recognition experiments, as given in paper:

> Marcus A. Angeloni and Helio Pedrini - "**Part-Based Representation and Classification for Face Recognition**", in proceedings of IEEE International Conference on Systems, Man, and Cybernetics (SMC 2016). Budapest, Hungary. p. 2900-2905

If you use this source code and/or its results, please cite our publication:

```
@inproceedings{Angeloni_SMC_2016, 
author={M. de Assis Angeloni and H. Pedrini}, 
title={Part-based representation and classification for face recognition}, 
year={2016},
month={Oct},
booktitle={2016 IEEE International Conference on Systems, Man, and Cybernetics (SMC)}, 
pages={002900-002905}, 
doi={10.1109/SMC.2016.7844680}, 
}
```

Dependencies
------------------

This code was tested to work under Python 2.7 on Ubuntu 14.04.

The required dependencies to run the experiments are `Numpy`, `SciPy`, `OpenCV`, `scikit-learn`, `scikit-image`, and `bob 1.2.2`.

To install the dependencies, run the following commands (need administration rights):

```
sudo apt-get install python-numpy python-scipy libopencv-dev python-opencv python-sklearn
sudo pip install Cython --upgrade
sudo pip install -U scikit-image
```

Furthermore, to install `bob 1.2.2` run the following commands to obtain and compile the code:
```
sudo add-apt-repository ppa:biometrics/bob
sudo apt-get update
sudo apt-get install wget git-core pkg-config cmake python-dev python-support liblapack-dev libatlas-base-dev libblitz1-dev libavformat-dev libavcodec-dev libswscale-dev libboost-all-dev libmatio-dev libjpeg8-dev libnetpbm10-dev libpng12-dev libtiff4-dev libgif-dev libhdf5-serial-dev libfftw3-dev texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended libsvm-dev libvl-dev dvipng dvipng
sudo apt-get install python-argparse python-matplotlib python-tornado python-sqlalchemy python-sphinx python-nose python-setuptools python-imaging ipython python-ipdb libqt4-core libqt4-dev libqt4-gui qt4-dev-tools

mkdir workdir
cd workdir
wget http://www.idiap.ch/software/bob/packages/bob-1.2.2.tar.gz
tar xvfz bob-1.2.2.tar.gz
cd bob-1.2.2
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sudo make
sudo make install
```

To finish the `bob` instalation, you might create a new file called `/etc/profile.d/local_python.sh` with the contents (need administration rights):

```
#!/bin/bash
export PYTHONPATH="<bob-build-path>/lib/python2.7/site-packages/":"$PYTHONPATH"
```

After that, you might add permission to this file:
```
chmod +x /etc/profile.d/local_python.sh
```

Finally, you do a logoff and a login, and the bob package will be available to your user.

Database and annotations
------------------

* Links to get the databases:

	* ARFACE database: http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html
		* 3,315 ppm files

	* XM2VTS database: http://www.ee.surrey.ac.uk/CVSSP/xm2vtsdb/
		* 2,592 ppm files

	* MUCT database: https://github.com/StephenMilborrow/muct
		* 3,755 jpg files

* Links to get the landmark annotations:

	* ARFACE database: http://cbcsl.ece.ohio-state.edu/AR_manual_markings.rar
		* 897 mat files

	* XM2VTS database: http://ibug.doc.ic.ac.uk/download/annotations/xm2vts.zip
		* 2,360 pts files

	* MUCT database: https://github.com/StephenMilborrow/muct/raw/master/muct-landmarks-v1.tar.gz
		* muct-landmarks/muct76-opencv.csv

Usage
------------------
