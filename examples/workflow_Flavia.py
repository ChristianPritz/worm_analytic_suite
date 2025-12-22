#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 15:12:08 2025

@author: christian
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
import pandas as pd
import cv2
from measurements import collect_annotations, analyze_annotations,analyze_thickness,analyze_staining_from_json, assign_conditions,organize_dataset
from worm_plotter import worm_width_plot, df_to_grouped_array,plot_grouped_values
from classifiers import  classify,run_kmeans_and_show,label_coco_areas,coco_areas_calculation,train_classifier
from annotation_tool_v8 import AnnotationTool



image_dir = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/' # this is where the images are sitting
point_csv =  '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_points.csv'
area_json =  "/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_areas.json"
color_csv =  '/media/my_device/space worms/makesenseai_analyzed_images/classes/class_colors.csv'

tool = AnnotationTool(image_dir,point_csv,area_json,color_csv)


# ##---------------------------------------------------------------------------##
# # OPTIONALLY: TRAIN THE LABELLING OF THE UNWORTHY..........................
# ##---------------------------------------------------------------------------##

# areas_json = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_areas.json'
# tlabels = label_coco_areas(areas_json)
# hists2 = coco_areas_calculation(areas_json)
# mdl = train_classifier(hists2,tlabels)
# predlabels,p = classify(hists2)
# np.sum(tlabels == predlabels)




