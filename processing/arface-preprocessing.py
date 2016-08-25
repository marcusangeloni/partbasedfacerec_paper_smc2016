#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Thu 7 Apr 2016 21:12:03

import sys
import os
import numpy
import scipy.io
import argparse
from datetime import datetime
import preprocess

annotation_extension       = "mat"
image_extension            = "ppm"
output_extension           = "npy"

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description='ARFace preprocessing')
parser.add_argument('annotation_dir', default='', help='Annotation directory')
parser.add_argument('image_dir', default='', help='Image directory')
parser.add_argument('output_dir', default='', help='Output directory')

args = parser.parse_args()

if (not(os.path.exists(args.annotation_dir))):
    print('Annotation directory (\"' + args.annotation_dir + '\") not found.')
    exit()

if (not(os.path.exists(args.image_dir))):
    print('Input image directory (\"' + args.image_dir + '\") not found.')
    exit()

if (not(os.path.exists(args.output_dir))):
    os.mkdir(args.output_dir)

annotation_dir = args.annotation_dir
image_dir = args.image_dir
output_dir = args.output_dir

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - ARFace - preprocessing started")
print("Annotation directory: " + annotation_dir)
print("Image directory: " + image_dir)
print("Output directory: " + output_dir)

# get the list of annotation files
userList = os.listdir(annotation_dir)

eyebrows_dir, eyes_dir, nose_dir, mouth_dir = preprocess.create_facial_parts_dir(output_dir)

for user in userList:
    if (user[len(user)-3:len(user)] != annotation_extension): # check the file extension
        continue

    # define the annotation file path
    annotation_path = os.path.join(annotation_dir, user)

    # replace the annotation file name in order to match with the corresponding image file
    user = user.replace("M", "m")
    user = user.replace("W", "w")
    
    # define the input image path
    user = user.replace(annotation_extension, image_extension)
    image_path = os.path.join(image_dir, user)
    
    # define the output image path
    user = user.replace(image_extension, output_extension)
    
    # load the input image
    image = preprocess.load_image(image_path)
    print (datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Current image " + image_path)
    
    # load the annotation file and read the coordinates
    mat_file = scipy.io.loadmat(annotation_path)
    coordinates = mat_file.get('faceCoordinates')
    
    angle = preprocess.get_angle(coordinates[7, 0], coordinates[7, 1], coordinates[14, 0], coordinates[14, 1])
    image,coordinates = preprocess.rotate_image_and_coordinates(image, coordinates, angle)
    
    # segment the eyes region
    coords = coordinates[0:26, 0:2]
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    eyes_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.03, 156, 29)
    
    # segment the eyebrows region
    coords = coordinates[26:50, 0:2]
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    eyebrows_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.03, 142, 20)
    
    # segment the nose region
    coords = numpy.concatenate((coordinates[50:52, 0:2], coordinates[55:76, 0:2]))
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    nose_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.15, 161, 114)
    
    # segment the mouth region
    coords = coordinates[83:109, 0:2]
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    mouth_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.08, 142, 82)
    
    # save the facial parts images
    numpy.save(os.path.join(eyebrows_dir, user), eyebrows_img)
    numpy.save(os.path.join(eyes_dir, user), eyes_img)
    numpy.save(os.path.join(nose_dir, user), nose_img)
    numpy.save(os.path.join(mouth_dir, user), mouth_img)

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - ARFace - preprocessing finished")