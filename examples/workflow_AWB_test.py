#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 14:24:11 2025

@author: wormulon
"""

import cv2
from measurements import awb
import matplotlib.pyplot as plt 

A = cv2.imread('/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/C_NG_G3_T2_310725rep2_20251021_180809_0.png')
B = cv2.imread('/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/C_LH_G3_T2_310725rep2_20251105_124450_6.png')
C = cv2.imread('/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/C_0G_G1_T2_250725rep1_20251006_154751_1.png')
D = cv2.imread('/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/C_0G_G1_T1_110725rep2_20251006_153930_1.png')


At = awb(A,quantile=0.05,debug=True)
fig,ax = plt.subplots(dpi=600)
ax.imshow(At[:,:,[2,1,0]])
plt.show()

Bt = awb(B,debug=True)
fig,ax = plt.subplots(dpi=600)
ax.imshow(Bt[:,:,[2,1,0]])
plt.show()

Ct = awb(C,quantile=0.05,debug=True)
fig,ax = plt.subplots(dpi=600)
ax.imshow(Ct[:,:,[2,1,0]])
plt.show()

Dt = awb(D,quantile=0.05,debug=True)
fig,ax = plt.subplots(dpi=600)
ax.imshow(Dt[:,:,[2,1,0]])
plt.show()


# convert all the images in the following path.................................

import os
import glob
import cv2

impath = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/'
savepath = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images_AWB/'

# create output directory if needed
os.makedirs(savepath, exist_ok=True)

# scan all image files in impath
image_list = sorted(
    glob.glob(os.path.join(impath, "*.[pjP][npNP][gG]")) +   # png / jpg / jpeg
    glob.glob(os.path.join(impath, "*.[tT][iI][fF]*"))       # tif / tiff
)

for i in image_list:
    img = cv2.imread(i)
    if img is None:
        print(f"Warning: could not read {i}")
        continue

    img_wb = awb(img, quantile=0.05, debug=False)

    # save under same basename
    basename = os.path.basename(i)
    outpath = os.path.join(savepath, basename)

    cv2.imwrite(outpath, img_wb)
    
    
