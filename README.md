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

* Training classifiers (3 datasets x 4 facial parts x 5 features x 3 classifiers):
```
python3 processing/classification.py arface_eyebrows_dct feat_arface/eyebrows/dct/ protocols/arface/ > log_arface_eyebrows_dct.txt
python3 processing/classification.py arface_eyebrows_gabor feat_arface/eyebrows/gabor/ protocols/arface/ > log_arface_eyebrows_gabor.txt
python3 processing/classification.py arface_eyebrows_glcm feat_arface/eyebrows/glcm/ protocols/arface/ > log_arface_eyebrows_glcm.txt
python3 processing/classification.py arface_eyebrows_hog feat_arface/eyebrows/hog/ protocols/arface/ > log_arface_eyebrows_hog.txt
python3 processing/classification.py arface_eyebrows_mlbp feat_arface/eyebrows/mlbp/ protocols/arface/ > log_arface_eyebrows_mlbp.txt
python3 processing/classification.py arface_eyes_dct feat_arface/eyes/dct/ protocols/arface/ > log_arface_eyes_dct.txt
python3 processing/classification.py arface_eyes_gabor feat_arface/eyes/gabor/ protocols/arface/ > log_arface_eyes_gabor.txt
python3 processing/classification.py arface_eyes_glcm feat_arface/eyes/glcm/ protocols/arface/ > log_arface_eyes_glcm.txt
python3 processing/classification.py arface_eyes_hog feat_arface/eyes/hog/ protocols/arface/ > log_arface_eyes_hog.txt
python3 processing/classification.py arface_eyes_mlbp feat_arface/eyes/mlbp/ protocols/arface/ > log_arface_eyes_mlbp.txt
python3 processing/classification.py arface_nose_dct feat_arface/nose/dct/ protocols/arface/ > log_arface_nose_dct.txt
python3 processing/classification.py arface_nose_gabor feat_arface/nose/gabor/ protocols/arface/ > log_arface_nose_gabor.txt
python3 processing/classification.py arface_nose_glcm feat_arface/nose/glcm/ protocols/arface/ > log_arface_nose_glcm.txt
python3 processing/classification.py arface_nose_hog feat_arface/nose/hog/ protocols/arface/ > log_arface_nose_hog.txt
python3 processing/classification.py arface_nose_mlbp feat_arface/nose/mlbp/ protocols/arface/ > log_arface_nose_mlbp.txt
python3 processing/classification.py arface_mouth_dct feat_arface/mouth/dct/ protocols/arface/ > log_arface_mouth_dct.txt
python3 processing/classification.py arface_mouth_gabor feat_arface/mouth/gabor/ protocols/arface/ > log_arface_mouth_gabor.txt
python3 processing/classification.py arface_mouth_glcm feat_arface/mouth/glcm/ protocols/arface/ > log_arface_mouth_glcm.txt
python3 processing/classification.py arface_mouth_hog feat_arface/mouth/hog/ protocols/arface/ > log_arface_mouth_hog.txt
python3 processing/classification.py arface_mouth_mlbp feat_arface/mouth/mlbp/ protocols/arface/ > log_arface_mouth_mlbp.txt

python3 processing/classification.py muct_eyebrows_dct feat_muct/eyebrows/dct/ protocols/muct/ > log_muct_eyebrows_dct.txt
python3 processing/classification.py muct_eyebrows_gabor feat_muct/eyebrows/gabor/ protocols/muct/ > log_muct_eyebrows_gabor.txt
python3 processing/classification.py muct_eyebrows_glcm feat_muct/eyebrows/glcm/ protocols/muct/ > log_muct_eyebrows_glcm.txt
python3 processing/classification.py muct_eyebrows_hog feat_muct/eyebrows/hog/ protocols/muct/ > log_muct_eyebrows_hog.txt
python3 processing/classification.py muct_eyebrows_mlbp feat_muct/eyebrows/mlbp/ protocols/muct/ > log_muct_eyebrows_mlbp.txt
python3 processing/classification.py muct_eyes_dct feat_muct/eyes/dct/ protocols/muct/ > log_muct_eyes_dct.txt
python3 processing/classification.py muct_eyes_gabor feat_muct/eyes/gabor/ protocols/muct/ > log_muct_eyes_gabor.txt
python3 processing/classification.py muct_eyes_glcm feat_muct/eyes/glcm/ protocols/muct/ > log_muct_eyes_glcm.txt
python3 processing/classification.py muct_eyes_hog feat_muct/eyes/hog/ protocols/muct/ > log_muct_eyes_hog.txt
python3 processing/classification.py muct_eyes_mlbp feat_muct/eyes/mlbp/ protocols/muct/ > log_muct_eyes_mlbp.txt
python3 processing/classification.py muct_nose_dct feat_muct/nose/dct/ protocols/muct/ > log_muct_nose_dct.txt
python3 processing/classification.py muct_nose_gabor feat_muct/nose/gabor/ protocols/muct/ > log_muct_nose_gabor.txt
python3 processing/classification.py muct_nose_glcm feat_muct/nose/glcm/ protocols/muct/ > log_muct_nose_glcm.txt
python3 processing/classification.py muct_nose_hog feat_muct/nose/hog/ protocols/muct/ > log_muct_nose_hog.txt
python3 processing/classification.py muct_nose_mlbp feat_muct/nose/mlbp/ protocols/muct/ > log_muct_nose_mlbp.txt
python3 processing/classification.py muct_mouth_dct feat_muct/mouth/dct/ protocols/muct/ > log_muct_mouth_dct.txt
python3 processing/classification.py muct_mouth_gabor feat_muct/mouth/gabor/ protocols/muct/ > log_muct_mouth_gabor.txt
python3 processing/classification.py muct_mouth_glcm feat_muct/mouth/glcm/ protocols/muct/ > log_muct_mouth_glcm.txt
python3 processing/classification.py muct_mouth_hog feat_muct/mouth/hog/ protocols/muct/ > log_muct_mouth_hog.txt
python3 processing/classification.py muct_mouth_mlbp feat_muct/mouth/mlbp/ protocols/muct/ > log_muct_mouth_mlbp.txt

python3 processing/classification.py xm2vts_eyebrows_dct feat_xm2vts/eyebrows/dct/ protocols/xm2vts/ > log_xm2vts_eyebrows_dct.txt
python3 processing/classification.py xm2vts_eyebrows_gabor feat_xm2vts/eyebrows/gabor/ protocols/xm2vts/ > log_xm2vts_eyebrows_gabor.txt
python3 processing/classification.py xm2vts_eyebrows_glcm feat_xm2vts/eyebrows/glcm/ protocols/xm2vts/ > log_xm2vts_eyebrows_glcm.txt
python3 processing/classification.py xm2vts_eyebrows_hog feat_xm2vts/eyebrows/hog/ protocols/xm2vts/ > log_xm2vts_eyebrows_hog.txt
python3 processing/classification.py xm2vts_eyebrows_mlbp feat_xm2vts/eyebrows/mlbp/ protocols/xm2vts/ > log_xm2vts_eyebrows_mlbp.txt
python3 processing/classification.py xm2vts_eyes_dct feat_xm2vts/eyes/dct/ protocols/xm2vts/ > log_xm2vts_eyes_dct.txt
python3 processing/classification.py xm2vts_eyes_gabor feat_xm2vts/eyes/gabor/ protocols/xm2vts/ > log_xm2vts_eyes_gabor.txt
python3 processing/classification.py xm2vts_eyes_glcm feat_xm2vts/eyes/glcm/ protocols/xm2vts/ > log_xm2vts_eyes_glcm.txt
python3 processing/classification.py xm2vts_eyes_hog feat_xm2vts/eyes/hog/ protocols/xm2vts/ > log_xm2vts_eyes_hog.txt
python3 processing/classification.py xm2vts_eyes_mlbp feat_xm2vts/eyes/mlbp/ protocols/xm2vts/ > log_xm2vts_eyes_mlbp.txt
python3 processing/classification.py xm2vts_nose_dct feat_xm2vts/nose/dct/ protocols/xm2vts/ > log_xm2vts_nose_dct.txt
python3 processing/classification.py xm2vts_nose_gabor feat_xm2vts/nose/gabor/ protocols/xm2vts/ > log_xm2vts_nose_gabor.txt
python3 processing/classification.py xm2vts_nose_glcm feat_xm2vts/nose/glcm/ protocols/xm2vts/ > log_xm2vts_nose_glcm.txt
python3 processing/classification.py xm2vts_nose_hog feat_xm2vts/nose/hog/ protocols/xm2vts/ > log_xm2vts_nose_hog.txt
python3 processing/classification.py xm2vts_nose_mlbp feat_xm2vts/nose/mlbp/ protocols/xm2vts/ > log_xm2vts_nose_mlbp.txt
python3 processing/classification.py xm2vts_mouth_dct feat_xm2vts/mouth/dct/ protocols/xm2vts/ > log_xm2vts_mouth_dct.txt
python3 processing/classification.py xm2vts_mouth_gabor feat_xm2vts/mouth/gabor/ protocols/xm2vts/ > log_xm2vts_mouth_gabor.txt
python3 processing/classification.py xm2vts_mouth_glcm feat_xm2vts/mouth/glcm/ protocols/xm2vts/ > log_xm2vts_mouth_glcm.txt
python3 processing/classification.py xm2vts_mouth_hog feat_xm2vts/mouth/hog/ protocols/xm2vts/ > log_xm2vts_mouth_hog.txt
python3 processing/classification.py xm2vts_mouth_mlbp feat_xm2vts/mouth/mlbp/ protocols/xm2vts/ > log_xm2vts_mouth_mlbp.txt
```

* Compute metrics of each experiment (3 datasets x 4 facial parts x 5 features x 3 classifiers):
```
python3 processing/experiment.py arface eyebrows dct_knn .
python3 processing/experiment.py arface eyebrows dct_rf .
python3 processing/experiment.py arface eyebrows dct_svm .
python3 processing/experiment.py arface eyebrows gabor_knn .
python3 processing/experiment.py arface eyebrows gabor_rf .
python3 processing/experiment.py arface eyebrows gabor_svm .
python3 processing/experiment.py arface eyebrows glcm_knn .
python3 processing/experiment.py arface eyebrows glcm_rf .
python3 processing/experiment.py arface eyebrows glcm_svm .
python3 processing/experiment.py arface eyebrows hog_knn .
python3 processing/experiment.py arface eyebrows hog_rf .
python3 processing/experiment.py arface eyebrows hog_svm .
python3 processing/experiment.py arface eyebrows mlbp_knn .
python3 processing/experiment.py arface eyebrows mlbp_rf .
python3 processing/experiment.py arface eyebrows mlbp_svm .
python3 processing/experiment.py arface eyes dct_knn .
python3 processing/experiment.py arface eyes dct_rf .
python3 processing/experiment.py arface eyes dct_svm .
python3 processing/experiment.py arface eyes gabor_knn .
python3 processing/experiment.py arface eyes gabor_rf .
python3 processing/experiment.py arface eyes gabor_svm .
python3 processing/experiment.py arface eyes glcm_knn .
python3 processing/experiment.py arface eyes glcm_rf .
python3 processing/experiment.py arface eyes glcm_svm .
python3 processing/experiment.py arface eyes hog_knn .
python3 processing/experiment.py arface eyes hog_rf .
python3 processing/experiment.py arface eyes hog_svm .
python3 processing/experiment.py arface eyes mlbp_knn .
python3 processing/experiment.py arface eyes mlbp_rf .
python3 processing/experiment.py arface eyes mlbp_svm .
python3 processing/experiment.py arface nose dct_knn .
python3 processing/experiment.py arface nose dct_rf .
python3 processing/experiment.py arface nose dct_svm .
python3 processing/experiment.py arface nose gabor_knn .
python3 processing/experiment.py arface nose gabor_rf .
python3 processing/experiment.py arface nose gabor_svm .
python3 processing/experiment.py arface nose glcm_knn .
python3 processing/experiment.py arface nose glcm_rf .
python3 processing/experiment.py arface nose glcm_svm .
python3 processing/experiment.py arface nose hog_knn .
python3 processing/experiment.py arface nose hog_rf .
python3 processing/experiment.py arface nose hog_svm .
python3 processing/experiment.py arface nose mlbp_knn .
python3 processing/experiment.py arface nose mlbp_rf .
python3 processing/experiment.py arface nose mlbp_svm .
python3 processing/experiment.py arface mouth dct_knn .
python3 processing/experiment.py arface mouth dct_rf .
python3 processing/experiment.py arface mouth dct_svm .
python3 processing/experiment.py arface mouth gabor_knn .
python3 processing/experiment.py arface mouth gabor_rf .
python3 processing/experiment.py arface mouth gabor_svm .
python3 processing/experiment.py arface mouth glcm_knn .
python3 processing/experiment.py arface mouth glcm_rf .
python3 processing/experiment.py arface mouth glcm_svm .
python3 processing/experiment.py arface mouth hog_knn .
python3 processing/experiment.py arface mouth hog_rf .
python3 processing/experiment.py arface mouth hog_svm .
python3 processing/experiment.py arface mouth mlbp_knn .
python3 processing/experiment.py arface mouth mlbp_rf .
python3 processing/experiment.py arface mouth mlbp_svm .

python3 processing/experiment.py muct eyebrows dct_knn .
python3 processing/experiment.py muct eyebrows dct_rf .
python3 processing/experiment.py muct eyebrows dct_svm .
python3 processing/experiment.py muct eyebrows gabor_knn .
python3 processing/experiment.py muct eyebrows gabor_rf .
python3 processing/experiment.py muct eyebrows gabor_svm .
python3 processing/experiment.py muct eyebrows glcm_knn .
python3 processing/experiment.py muct eyebrows glcm_rf .
python3 processing/experiment.py muct eyebrows glcm_svm .
python3 processing/experiment.py muct eyebrows hog_knn .
python3 processing/experiment.py muct eyebrows hog_rf .
python3 processing/experiment.py muct eyebrows hog_svm .
python3 processing/experiment.py muct eyebrows mlbp_knn .
python3 processing/experiment.py muct eyebrows mlbp_rf .
python3 processing/experiment.py muct eyebrows mlbp_svm .
python3 processing/experiment.py muct eyes dct_knn .
python3 processing/experiment.py muct eyes dct_rf .
python3 processing/experiment.py muct eyes dct_svm .
python3 processing/experiment.py muct eyes gabor_knn .
python3 processing/experiment.py muct eyes gabor_rf .
python3 processing/experiment.py muct eyes gabor_svm .
python3 processing/experiment.py muct eyes glcm_knn .
python3 processing/experiment.py muct eyes glcm_rf .
python3 processing/experiment.py muct eyes glcm_svm .
python3 processing/experiment.py muct eyes hog_knn .
python3 processing/experiment.py muct eyes hog_rf .
python3 processing/experiment.py muct eyes hog_svm .
python3 processing/experiment.py muct eyes mlbp_knn .
python3 processing/experiment.py muct eyes mlbp_rf .
python3 processing/experiment.py muct eyes mlbp_svm .
python3 processing/experiment.py muct nose dct_knn .
python3 processing/experiment.py muct nose dct_rf .
python3 processing/experiment.py muct nose dct_svm .
python3 processing/experiment.py muct nose gabor_knn .
python3 processing/experiment.py muct nose gabor_rf .
python3 processing/experiment.py muct nose gabor_svm .
python3 processing/experiment.py muct nose glcm_knn .
python3 processing/experiment.py muct nose glcm_rf .
python3 processing/experiment.py muct nose glcm_svm .
python3 processing/experiment.py muct nose hog_knn .
python3 processing/experiment.py muct nose hog_rf .
python3 processing/experiment.py muct nose hog_svm .
python3 processing/experiment.py muct nose mlbp_knn .
python3 processing/experiment.py muct nose mlbp_rf .
python3 processing/experiment.py muct nose mlbp_svm .
python3 processing/experiment.py muct mouth dct_knn .
python3 processing/experiment.py muct mouth dct_rf .
python3 processing/experiment.py muct mouth dct_svm .
python3 processing/experiment.py muct mouth gabor_knn .
python3 processing/experiment.py muct mouth gabor_rf .
python3 processing/experiment.py muct mouth gabor_svm .
python3 processing/experiment.py muct mouth glcm_knn .
python3 processing/experiment.py muct mouth glcm_rf .
python3 processing/experiment.py muct mouth glcm_svm .
python3 processing/experiment.py muct mouth hog_knn .
python3 processing/experiment.py muct mouth hog_rf .
python3 processing/experiment.py muct mouth hog_svm .
python3 processing/experiment.py muct mouth mlbp_knn .
python3 processing/experiment.py muct mouth mlbp_rf .
python3 processing/experiment.py muct mouth mlbp_svm .

python3 processing/experiment.py xm2vts eyebrows dct_knn .
python3 processing/experiment.py xm2vts eyebrows dct_rf .
python3 processing/experiment.py xm2vts eyebrows dct_svm .
python3 processing/experiment.py xm2vts eyebrows gabor_knn .
python3 processing/experiment.py xm2vts eyebrows gabor_rf .
python3 processing/experiment.py xm2vts eyebrows gabor_svm .
python3 processing/experiment.py xm2vts eyebrows glcm_knn .
python3 processing/experiment.py xm2vts eyebrows glcm_rf .
python3 processing/experiment.py xm2vts eyebrows glcm_svm .
python3 processing/experiment.py xm2vts eyebrows hog_knn .
python3 processing/experiment.py xm2vts eyebrows hog_rf .
python3 processing/experiment.py xm2vts eyebrows hog_svm .
python3 processing/experiment.py xm2vts eyebrows mlbp_knn .
python3 processing/experiment.py xm2vts eyebrows mlbp_rf .
python3 processing/experiment.py xm2vts eyebrows mlbp_svm .
python3 processing/experiment.py xm2vts eyes dct_knn .
python3 processing/experiment.py xm2vts eyes dct_rf .
python3 processing/experiment.py xm2vts eyes dct_svm .
python3 processing/experiment.py xm2vts eyes gabor_knn .
python3 processing/experiment.py xm2vts eyes gabor_rf .
python3 processing/experiment.py xm2vts eyes gabor_svm .
python3 processing/experiment.py xm2vts eyes glcm_knn .
python3 processing/experiment.py xm2vts eyes glcm_rf .
python3 processing/experiment.py xm2vts eyes glcm_svm .
python3 processing/experiment.py xm2vts eyes hog_knn .
python3 processing/experiment.py xm2vts eyes hog_rf .
python3 processing/experiment.py xm2vts eyes hog_svm .
python3 processing/experiment.py xm2vts eyes mlbp_knn .
python3 processing/experiment.py xm2vts eyes mlbp_rf .
python3 processing/experiment.py xm2vts eyes mlbp_svm .
python3 processing/experiment.py xm2vts nose dct_knn .
python3 processing/experiment.py xm2vts nose dct_rf .
python3 processing/experiment.py xm2vts nose dct_svm .
python3 processing/experiment.py xm2vts nose gabor_knn .
python3 processing/experiment.py xm2vts nose gabor_rf .
python3 processing/experiment.py xm2vts nose gabor_svm .
python3 processing/experiment.py xm2vts nose glcm_knn .
python3 processing/experiment.py xm2vts nose glcm_rf .
python3 processing/experiment.py xm2vts nose glcm_svm .
python3 processing/experiment.py xm2vts nose hog_knn .
python3 processing/experiment.py xm2vts nose hog_rf .
python3 processing/experiment.py xm2vts nose hog_svm .
python3 processing/experiment.py xm2vts nose mlbp_knn .
python3 processing/experiment.py xm2vts nose mlbp_rf .
python3 processing/experiment.py xm2vts nose mlbp_svm .
python3 processing/experiment.py xm2vts mouth dct_knn .
python3 processing/experiment.py xm2vts mouth dct_rf .
python3 processing/experiment.py xm2vts mouth dct_svm .
python3 processing/experiment.py xm2vts mouth gabor_knn .
python3 processing/experiment.py xm2vts mouth gabor_rf .
python3 processing/experiment.py xm2vts mouth gabor_svm .
python3 processing/experiment.py xm2vts mouth glcm_knn .
python3 processing/experiment.py xm2vts mouth glcm_rf .
python3 processing/experiment.py xm2vts mouth glcm_svm .
python3 processing/experiment.py xm2vts mouth hog_knn .
python3 processing/experiment.py xm2vts mouth hog_rf .
python3 processing/experiment.py xm2vts mouth hog_svm .
python3 processing/experiment.py xm2vts mouth mlbp_knn .
python3 processing/experiment.py xm2vts mouth mlbp_rf .
python3 processing/experiment.py xm2vts mouth mlbp_svm .
```