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

This code was tested to work under Python 3.7 on Ubuntu 19.04.

The required dependencies to run the experiments are `Numpy`, `SciPy`, `OpenCV`, `scikit-learn`, `scikit-image`, and `bob`.

To install the dependencies, run the following commands (need administration rights):

```
sudo apt-get install python-numpy python-scipy libopencv-dev python-opencv python-sklearn
sudo pip install Cython --upgrade
sudo pip install -U scikit-image
```

Furthermore, to install `bob`, install miniconda and run the following commands (based on https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob/doc/install.html):
```
conda update -n base conda
conda config --set show_channel_urls True

conda create --name bob_py3 --override-channels \
  -c https://www.idiap.ch/software/bob/conda -c defaults \
  python=3 bob
conda activate bob_py3
conda config --env --add channels defaults
conda config --env --add channels https://www.idiap.ch/software/bob/conda

conda install bob.io.image bob.bio.base bob.math bob.measure bob.bio.face bob.core bob.fusion.base bob.ip.base bob.ip.color bob.ip.dlib bob.ip.facedetect bob.ip.facelandmarks bob.ip.gabor

```


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

* Plot facial landmarks [Optional]:
```
python3 utils/arface-plotLandmarks.py annotations/arface_annot/AR_manual_markings/ images/arface_imgs out_arface
python3 utils/muct-plotLandmarks.py annotations/muct_annot/muct-landmarks/muct76-opencv.csv images/muct_imgs/jpg/ out_muct
python3 utils/xm2vts-plotLandmarks.py annotations/xm2vts_annot/landmarks/ images/xm2vts_imgs/imgs out_xm2vts
```

* Preprocessing:
```
python3 processing/arface-preprocessing.py annotations/arface_annot/AR_manual_markings/ images/arface_imgs/ processed_arface
python3 processing/muct-preprocessing.py annotations/muct_annot/muct-landmarks/muct76-opencv.csv images/muct_imgs/jpg/ processed_muct
python3 processing/xm2vts-preprocessing.py annotations/xm2vts_annot/landmarks/ images/xm2vts_imgs/imgs/ processed_xm2vts
```

* Feature extraction:
```
python3 processing/feature_extraction.py processed_arface feat_arface all
python3 processing/feature_extraction.py processed_muct/ feat_muct all
python3 processing/feature_extraction.py processed_xm2vts/ feat_xm2vts all
```

