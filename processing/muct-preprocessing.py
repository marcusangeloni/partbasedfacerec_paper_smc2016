#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Thu 7 Apr 2016 21:12:03

import sys
import os
import numpy
import argparse
from datetime import datetime
import preprocess

image_extension            = ".jpg"
output_extension           = ".npy"

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description='MUCT preprocessing')
parser.add_argument('annotation_path', default='', help='Annotation file')
parser.add_argument('image_dir', default='', help='Image directory')
parser.add_argument('output_dir', default='', help='Output directory')

args = parser.parse_args()

if (not(os.path.exists(args.annotation_path))):
    print('Annotation path (\"' + args.annotation_path + '\") not found.')
    exit()
elif (not(os.path.isfile(args.annotation_path))):
    print('Annotation path (\"' + args.annotation_path + '\") is not a file.')
    exit()

if (not(os.path.exists(args.image_dir))):
    print('Input image directory (\"' + args.image_dir + '\") not found.')
    exit()

if (not(os.path.exists(args.output_dir))):
    os.mkdir(args.output_dir)

annotation_path = args.annotation_path
image_dir = args.image_dir
output_dir = args.output_dir

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - MUCT - preprocessing started")
print("Annotation file: " + annotation_path)
print("Image directory: " + image_dir)
print("Output directory: " + output_dir)

# read the annotation file (unique for this dataset)
coord_file = open(annotation_path, 'r')
lines = coord_file.readlines()
coord_file.close()

eyebrows_dir, eyes_dir, nose_dir, mouth_dir = preprocess.create_facial_parts_dir(output_dir)

# start in the second line due to the first one is the file header
for user_idx in range(1, 3756):
    line = lines[user_idx].rstrip()
    fields = line.split(',')
    user = fields[0] + image_extension # first column has the image file
    
    # define the input image path
    image_path = os.path.join(image_dir, user)

    # define the output image path
    user = user.replace(image_extension, output_extension)
    
    # load the input image
    image = preprocess.load_image(image_path)
    print (datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Current image " + image_path)

    # read landmark annotations
    coordinates = numpy.zeros((76,2), numpy.float)

    for coord_idx in range(1, 77):
        coordinates[coord_idx - 1, 0] = float(fields[coord_idx * 2])
        coordinates[coord_idx - 1, 1] = float(fields[coord_idx * 2 + 1])
   
    angle = preprocess.get_angle(coordinates[29, 0], coordinates[29, 1], coordinates[34, 0], coordinates[34, 1])
    image,coordinates = preprocess.rotate_image_and_coordinates(image, coordinates, angle)

    # segment the eyes region
    coords = coordinates[27:37, 0:2]
    coords = numpy.concatenate((coords, coordinates[68:76, 0:2]), axis = 0)
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    eyes_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.03, 156, 29)

    # segment the eyebrows region
    coords = coordinates[15:27, 0:2]
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    eyebrows_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.05, 142, 20)

    # segment the nose region
    coords = coordinates[38:45, 0:2]
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    nose_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.18, 161, 114)

    # segment the mouth region
    coords = coordinates[48:67, 0:2]
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    mouth_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.08, 142, 82)

    # save the facial parts images
    numpy.save(os.path.join(eyebrows_dir, user), eyebrows_img)
    numpy.save(os.path.join(eyes_dir, user), eyes_img)
    numpy.save(os.path.join(nose_dir, user), nose_img)
    numpy.save(os.path.join(mouth_dir, user), mouth_img)
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - MUCT - preprocessing finished")