#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Thu 7 Apr 2016 21:12:03

import os
import scipy.io
import cv2
import numpy
import argparse
from datetime import datetime

annotation_extension       = "mat"
image_extension            = "ppm"
output_extension           = "jpg"

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description='Plot facial landmarks in the ARFace database')
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

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - ARFace - plot facial landmarks started")
print("Annotation directory: " + annotation_dir)
print("Image directory: " + image_dir)
print("Output directory: " + output_dir)


# get the list of annotation files
userList = os.listdir(annotation_dir)

for user in userList:
    if(user[len(user)-3:len(user)] != annotation_extension): # check the file extension
        continue

    annotation_path = os.path.join(annotation_dir, user) # define the annotation file path

    # replace the annotation file name in order to match with the corresponding image file
    user = user.replace("M", "m")
    user = user.replace("W", "w")

    # define the input image path
    user = user.replace(annotation_extension, image_extension)
    image_path = os.path.join(image_dir, user)

    # define the output image path
    user = user.replace(image_extension, output_extension)
    output_path = os.path.join(output_dir, user)

    # load the input image
    image = cv2.imread(image_path)
    print (datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Current image " + output_path)

    # load the annotation file and read the coordinates
    mat_file = scipy.io.loadmat(annotation_path)
    coordinates = mat_file.get('faceCoordinates')
    
    coord_idx = 1
    
    for c in coordinates:

        # select a color for each facial part

        # 1-26 eyes
        if (coord_idx <= 26):
            color = (0, 255, 0) # green

        # 27-50 eyebrows
        elif (coord_idx <= 50): 
            color = (255, 255, 0) # cyan

        # 51-83 nose
        elif (coord_idx <= 83):
            color = (0, 0, 255) # red

        # 84-109 mouth
        elif (coord_idx <= 109):
            color = (0, 255, 255) # yellow

        # 110-130 face contour
        elif (coord_idx <= 130):
            color = (255, 0, 0) # blue

        # plot the landmark coordinates
        cv2.circle(image, (c[0].astype(numpy.uint32), c[1].astype(numpy.uint32)), 1, color, -1)
        coord_idx += 1

    cv2.imwrite(output_path, image) # save the output image

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - ARFace - plot facial landmarks finished")