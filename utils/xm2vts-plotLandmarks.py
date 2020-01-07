#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Thu 7 Apr 2016 21:12:03

import os
import cv2
import numpy
import argparse
from datetime import datetime

annotation_extension       = "pts"
image_extension            = "ppm"
output_extension           = "jpg"

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description='Plot facial landmarks in the XM2VTS database')
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

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - XM2VTS - plot facial landmarks started")
print("Annotation directory: " + annotation_dir)
print("Image directory: " + image_dir)
print("Output directory: " + output_dir)


# get the list of annotation files
userList = os.listdir(annotation_dir)

for user in userList:
    if(user[len(user)-3:len(user)] != annotation_extension or user.startswith('.')): # check the file extension
        continue

    annotation_path = os.path.join(annotation_dir, user) # define the annotation file path

    # define the output image path
    user = user.replace(annotation_extension, output_extension)
    output_path = os.path.join(output_dir, user)

    # define the input image path
    user = user.replace(output_extension, image_extension)
    image_path = os.path.join(image_dir, user[0:3])
    image_path = os.path.join(image_path, user)

    # load the input image
    image = cv2.imread(image_path)
    print (datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Current image " + output_path)

    # load the annotation file and read the coordinates
    coord_file = open(annotation_path, 'r')
    lines = coord_file.readlines()
    coord_file.close()
            
    for idx in range(3, 71):
        coord_idx = idx - 2

        # select a color for each facial part

        # 1-17 face contour
        if (coord_idx <= 17):
            color = (255, 0, 0) # blue

        # 18-27 eyebrows
        elif (coord_idx <= 27): 
            color = (255, 255, 0) # cyan

        # 28-36 nose
        elif (coord_idx <= 36):
            color = (0, 0, 255) # red

        # 37-48 eyes
        elif (coord_idx <= 48):
            color = (0, 255, 0) # green

        # 49-68 mouth
        elif (coord_idx <= 68):
            color = (0, 255, 255) # yellow

        # plot the landmark coordinates
        x = int(round(float(lines[idx].split()[0])))
        y = int(round(float(lines[idx].split()[1])))

        cv2.circle(image, (x, y), 1, color, -1)

    cv2.imwrite(output_path, image) # save the output image

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - XM2VTS - plot facial landmarks finished")
