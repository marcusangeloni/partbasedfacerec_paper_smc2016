#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Thu 7 Apr 2016 21:12:03

import os
import cv2
import numpy
import argparse
from datetime import datetime

image_extension = ".jpg"

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description='Plot facial landmarks in the MUCT database')
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

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - MUCT - plot facial landmarks started")
print("Annotation file: " + annotation_path)
print("Image directory: " + image_dir)
print("Output directory: " + output_dir)


# read the annotation file (unique for this dataset)
coord_file = open(annotation_path, 'r')
lines = coord_file.readlines()
coord_file.close()

# start in the second line due to the first one is the file header
for user_idx in range(1, 3756):
    line = lines[user_idx].rstrip()
    fields = line.split(',')
    user = fields[0] + image_extension # first column has the image file
    
    # define the input image path
    image_path = os.path.join(image_dir, user)
    
    # define the output image path
    output_path = os.path.join(output_dir, user)

    # load the input image
    image = cv2.imread(image_path)
    print (datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Current image " + output_path)

    for coord_idx in range(1, 77):

        # select a color for each facial part

        # 1-15 face contour
        if (coord_idx <= 15):
            color = (255, 0, 0) # blue

        # 16-27 eyebrows
        elif (coord_idx <= 27):
            color = (255, 255, 0) # cyan

        # 28-37 and 69-76 eyes
        elif (coord_idx <= 37) or ((coord_idx >= 69) and (coord_idx <= 76)):
            color = (0, 255, 0) # green

        # 38-48 and 68 nose
        elif (coord_idx <= 48) or (coord_idx == 68):
            color = (0, 0, 255) # red

        # 49-67 mouth
        elif (coord_idx <= 67):
            color = (0, 255, 255) # yellow

        # plot the landmark coordinates
        x = int(round(float(fields[coord_idx * 2])))
        y = int(round(float(fields[coord_idx * 2 + 1])))
        cv2.circle(image, (x, y), 1, color, -1)

    cv2.imwrite(output_path, image) # save the output image

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - MUCT - plot facial landmarks finished")