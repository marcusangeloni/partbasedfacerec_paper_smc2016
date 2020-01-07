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

annotation_extension       = "pts"
image_extension            = "ppm"
output_extension           = "npy"

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description='XM2VTS preprocessing')
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

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - XM2VTS - preprocessing started")
print("Annotation directory: " + annotation_dir)
print("Image directory: " + image_dir)
print("Output directory: " + output_dir)

# get the list of annotation files
userList = os.listdir(annotation_dir)

eyebrows_dir, eyes_dir, nose_dir, mouth_dir = preprocess.create_facial_parts_dir(output_dir)

for user in userList:
    if (user[len(user)-3:len(user)] != annotation_extension or user.startswith('.')): # check the file extension
        continue

    # define the annotation file path
    annotation_path = os.path.join(annotation_dir, user)

    # define the input image path
    user = user.replace(annotation_extension, image_extension)
    image_path = os.path.join(image_dir, user[0:3])
    image_path = os.path.join(image_path, user)

    # define the output image path
    user = user.replace(image_extension, output_extension)
    image = preprocess.load_image(image_path)
    print (datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Current image " + image_path)

    # read landmark annotations
    coord_file = open(annotation_path, 'r')
    lines = coord_file.readlines()
    coord_file.close()

    coordinates = numpy.zeros((68,2), numpy.float)
            
    for idx in range(3, 71):
        line = lines[idx].split()
        coordinates[idx - 3, 0] = float(line[0])
        coordinates[idx - 3, 1] = float(line[1])

    angle = preprocess.get_angle(coordinates[39, 0], coordinates[39, 1], coordinates[42, 0], coordinates[42, 1])
    image,coordinates = preprocess.rotate_image_and_coordinates(image, coordinates, angle)

    # segment the eyes region
    coords = coordinates[36:48, 0:2]
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    eyes_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.03, 156, 29)

    # segment the eyebrows region
    coords = coordinates[17:27, 0:2]
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    eyebrows_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.03, 142, 20)

    # segment the nose region
    coords = coordinates[29:36, 0:2]
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    nose_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.4, 161, 114)

    # segment the mouth region
    coords = coordinates[48:68, 0:2]
    minX, maxX, minY, maxY = preprocess.find_min_max(coords)
    mouth_img = preprocess.crop_and_resize(image, minX, maxX, minY, maxY, 0.08, 142, 82)

    # save the facial parts images
    numpy.save(os.path.join(eyebrows_dir, user), eyebrows_img)
    numpy.save(os.path.join(eyes_dir, user), eyes_img)
    numpy.save(os.path.join(nose_dir, user), nose_img)
    numpy.save(os.path.join(mouth_dir, user), mouth_img)
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - XM2VTS - preprocessing finished")
