#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Thu 7 Apr 2016 21:12:03

import numpy
import bob
import os
import sys
import math
import skimage.feature
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import itemfreq
import argparse
from datetime import datetime

# compute tan and triggs illumination
def compute_tanTriggs(image):
    tt = bob.ip.TanTriggs(0.2, 1., 2., 5, 10., 0.1)
    data = tt(image)
    return data

# compute gray-level co-occurrence matrix descriptor
def compute_glcm(image):
    glcm = greycomatrix(image, distances=[1, 2, 3, 4], angles=[0, numpy.pi/4, numpy.pi/2, 3*numpy.pi/4], levels=256, symmetric=False, normed=True)
    
    features = numpy.zeros((6, glcm.shape[2] * glcm.shape[3]), dtype = numpy.float64)
    features[0, :] = greycoprops(glcm, 'contrast').flatten()
    features[1, :] = greycoprops(glcm, 'energy').flatten()
    features[2, :] = greycoprops(glcm, 'homogeneity').flatten()
    features[3, :] = greycoprops(glcm, 'dissimilarity').flatten()
    features[4, :] = greycoprops(glcm, 'correlation').flatten()
    features[5, :] = greycoprops(glcm, 'ASM').flatten()
    
    return features

# compute 2D Discrete Cosine Transform descriptor
def compute_dct(image):
    # block = 12x12, overlap = 11x11, normalized block, normalized features
    dct = bob.ip.DCTFeatures(12, 12, 11, 11, 45, True, True)
    features = dct(image)
    return features

# compute local binary patterns
def compute_lbp(image, P, R, blockSize, overlap):
    stepY = blockSize[0] - overlap[0]
    stepX = blockSize[1] - overlap[1]
    
    bins = P * (P - 1) + 3
    
    lbp = bob.ip.LBP(P, R, R, True, False, False, True, False) # circular, uniform pattern
    lbpImage = lbp(image)
    
    nBlocksY = (lbpImage.shape[0] - overlap[0]) / stepY
    nBlocksX = (lbpImage.shape[1] - overlap[1]) / stepX
    totalBlocks = nBlocksY * nBlocksX
    
    features = numpy.zeros((totalBlocks, bins), dtype = numpy.float64)
    
    x1 = 0
    x2 = blockSize[1]
    y1 = 0
    y2 = blockSize[0]
    idx = 0
    
    while (y2 < lbpImage.shape[0]):
        while (x2 < lbpImage.shape[1]):
            block = lbpImage[y1:y2, x1:x2]
            values = itemfreq(block.ravel())
            
            for v in values:
                features[idx, int(v[0])] = v[1]
            
            x1 += stepX
            x2 += stepX
            idx += 1
        
        y1 += stepY
        y2 += stepY
    
    return features

# compute multiscale local binary pattern descriptor
def compute_mlbp(image, P, blockSize, overlap):
    # R = 1
    features_r1 = compute_lbp(image[2:image.shape[0]-2, 2:image.shape[1]-2], P, 1, blockSize, overlap) # to get the same dimension of radius = 3
    
    # R = 3
    features_r3 = compute_lbp(image, P, 3, blockSize, overlap)
    
    features = numpy.concatenate((features_r1, features_r3), axis=1)
    return features

# compute histogram of oriented gradients
def compute_hog(image, pixels_cell):
    features = skimage.feature.hog(image, orientations=9, pixels_per_cell=pixels_cell, cells_per_block=(1, 1))
    return features

# compute histogram of oriented gradients descriptor
def compute_hog_descriptor(image):
    # blocos de 4x4
    features_b4 = compute_hog(image, (4, 4))
    
    #bloco de 8x8
    features_b8 = compute_hog(image, (8, 8))
    
    features = numpy.concatenate((features_b4, features_b8))
    return features

# compute gabor wavelet descriptor
def compute_gabor(image):
    gwt = bob.ip.GaborWaveletTransform(number_of_scales = 5,
                                       number_of_angles = 8,
                                       sigma = 2 * math.pi,
                                       k_max = math.pi / 2.,
                                       k_fac = math.sqrt(.5),
                                       pow_of_k = 0,
                                       dc_free = True)
                                       
    image_c = image.astype(numpy.complex128)
    trafo_image = gwt.empty_trafo_image(image_c)
    gwt(image_c, trafo_image)
    features = numpy.abs(trafo_image)
                                       
    return features


#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description='Feature extraction')
parser.add_argument('image_dir', default='', help='Image directory root (with facial parts folders and npy files)')
parser.add_argument('output_dir', default='', help='Output directory with extracted features')
parser.add_argument('features', default='', help='Features to be extracted [all, dct, mlbp, hog, gabor, glcm]')

args = parser.parse_args()

if (not(os.path.exists(args.image_dir))):
    print('Image directory root (\"' + args.image_dir + '\") not found.')
    exit()

if (not(os.path.exists(args.output_dir))):
    os.mkdir(args.output_dir)

if not(args.features in ['all', 'dct', 'mlbp', 'hog', 'gabor', 'glcm']):
    print('Features not implemented (\"' + args.image_dir + '\"). Available: [all, dct, mlbp, hog, gabor, glcm]')
    exit()

image_dir = args.image_dir
output_dir = args.output_dir
features = args.features

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - feature extraction started")
print("Image directory: " + image_dir)
print("Output directory: " + output_dir)
print("Selected features: " + features)

partList = os.listdir(image_dir)

for part in partList:

    in_part_dir = os.path.join(image_dir, part)
    print (in_part_dir)
    if (not(os.path.isdir(in_part_dir))):
        continue

    print (datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Current facial part " + part)
    out_part_dir = os.path.join(output_dir, part)

    # create feature directories
    if (not(os.path.exists(out_part_dir))):
        os.mkdir(out_part_dir)

    if (features in ['all', 'dct']):
        dct_dir = os.path.join(out_part_dir, 'dct')
        if (not(os.path.exists(dct_dir))):
            os.mkdir(dct_dir)

    if (features in ['all', 'mlbp']):
        mlbp_dir = os.path.join(out_part_dir, 'mlbp')
        if (not(os.path.exists(mlbp_dir))):
            os.mkdir(mlbp_dir)

    if (features in ['all', 'hog']):
        hog_dir = os.path.join(out_part_dir, 'hog')
        if (not(os.path.exists(hog_dir))):
            os.mkdir(hog_dir)

    if (features in ['all', 'gabor']):
        gabor_dir = os.path.join(out_part_dir, 'gabor')
        if (not(os.path.exists(gabor_dir))):
            os.mkdir(gabor_dir)

    if (features in ['all', 'glcm']):
        glcm_dir = os.path.join(out_part_dir, 'glcm')
        if (not(os.path.exists(glcm_dir))):
            os.mkdir(glcm_dir)

    images = os.listdir(in_part_dir)

    for i in images:
        
        if (i[len(i)-3:len(i)] != 'npy'): # check the file extension
            continue
        
        print (datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Current file " + i)
        img = numpy.load(os.path.join(in_part_dir, i))
        tt_img = compute_tanTriggs(img)

        if (features in ['all', 'dct']):
            descriptor = compute_dct(tt_img)
            numpy.save(os.path.join(dct_dir, i), descriptor)

        if (features in ['all', 'gabor']):
            descriptor = compute_gabor(tt_img)
            numpy.save(os.path.join(gabor_dir, i), descriptor)

        if (features in ['all', 'glcm']):
            descriptor = compute_glcm(img) # without tan and triggs
            numpy.save(os.path.join(glcm_dir, i), descriptor)

        if (features in ['all', 'hog']):
            # remove border according to each facial
            if (part =='eyebrows'):
                img_hog = tt_img[2:18, 3:139]
            elif (part =='eyes'):
                img_hog = tt_img[3:27, 2:154]
            elif (part == 'nose'):
                img_hog = tt_img[1:113, 1:161]
            elif (part == 'mouth'):
                img_hog = tt_img[1:81, 3:139]
            else:
                img_hog = None
                print('HOG not calculated due to facial part is unknown (' + part + ')')

            if not(img_hog is None):
                descriptor = compute_hog_descriptor(img_hog)
                numpy.save(os.path.join(hog_dir, i), descriptor)

        if (features in ['all', 'mlbp']):
            # neighbors, block size and overlap according to each facial part
            if (part in ['eyebrows', 'eyes']):
                descriptor = compute_mlbp(tt_img, 8, (8, 8), (4, 4))
            elif (part == 'nose'):
                 descriptor = compute_mlbp(tt_img, 8, (16, 16), (8, 8))
            elif (part == 'mouth'):
                descriptor = compute_mlbp(tt_img, 4, (16, 16), (8, 8))
            else:
                descriptor = None
                print('MLBP not calculated due to facial part is unknown (' + part + ')')

            if not(descriptor is None):
                numpy.save(os.path.join(mlbp_dir, i), descriptor)

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - feature extraction finished")