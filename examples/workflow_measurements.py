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


##---------------------------------------------------------------------------##
#              M E R G E    T H E    A N N O T A T I O N S
#
# These lines of code pool all the annotations and images into a single 
# locations (output_dir, image_dir). And also pools all the annotations into 
# single annotation files.
# this scans the entire directory tree and puts all the files together 
# be cautious of doubling entries 
##---------------------------------------------------------------------------##


## this copies all the files together into a single image directory 
## and a single labels directory (output)

organize_dataset("/media/my_device/space worms/makesenseai_analyzed_images/to analyze")



##---------------------------------------------------------------------------##
## this joins the labels and throws out bad labels. 

settings = {"rim_score_cutoff":0.05,"debug":True,"use_classifier":False}
collect_annotations(settings,
    anno_dir='/media/my_device/space worms/makesenseai_analyzed_images/to analyze/labels', # this is where there raw annoations go
    image_dir='/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images', # this is where the images are sitting
    color_csv='/media/my_device/space worms/makesenseai_analyzed_images/classes/class_colors.csv', 
    output_dir='/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output' #This is where the joined labels go 
)


##---------------------------------------------------------------------------##
#              C H E CK    T H E    A N N O T A T I O N S
#
# This allows to change the annotations of animals and the shapes
#
##---------------------------------------------------------------------------##

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


##---------------------------------------------------------------------------##
#             A N A L Y Z E    T H E    S H A P E S
##---------------------------------------------------------------------------##

df = analyze_annotations(
    areas_json='/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_areas.json',
    points_csv='/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_points.csv',
    image_dir='/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/',
    settings=settings,
    analysis_func=analyze_thickness
)


##---------------------------------------------------------------------------##
#
#             A N A L Y Z E    C O L O R    C H A N N E L S
# 
# This performs a color deconvolution of the images and 
# quantifies the color in the annotation 
# it also exports the images of straightened animals. The quantification 
# however is performed on the original curved animal (original label, 
# original image) 
##---------------------------------------------------------------------------##

color_matrix = np.array([
    [0.2117, 0.6481, 0.7299],  # R
    [0.5396, 0.5686, 0.6182],  # G
    [0.6250, 0.5654, 0.5375]   # B
])

color_matrix = color_matrix.transpose()
settings = {"rim_score_cutoff":0.09,
            "imsize":3120}


df = analyze_staining_from_json('/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_areas.json',
    '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/',color_matrix,settings,
    output_dir = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/output/',
    df=df)

##---------------------------------------------------------------------------##
#                 A S S I G N I N G   L A B E L S 
##---------------------------------------------------------------------------##

df2 =assign_conditions(df,col_name ="image_name_y")


##---------------------------------------------------------------------------##
#                 Plotting
##---------------------------------------------------------------------------##

cols = ['percent_0.05', 'percent_0.1', 'percent_0.15', 'percent_0.2',
       'percent_0.25', 'percent_0.3', 'percent_0.35', 'percent_0.4',
       'percent_0.45', 'percent_0.5', 'percent_0.55', 'percent_0.6',
       'percent_0.65', 'percent_0.7', 'percent_0.75', 'percent_0.8',
       'percent_0.85', 'percent_0.9', 'percent_0.95',]

percents = np.arange(0.05,1,0.05)


from worm_plotter import worm_width_plot, filter_dataframe


df4 = filter_dataframe(df, ["is_NG","generation","label_id"], [1,3,[4,5,9,8]])
df5 = filter_dataframe(df, ["is_0G","generation","label_id"], [1,3,[4,5,9,8]])

df_list = [df4,df5]

flips = [False,False]
colors = np.array([[0.05,0.15,0.95],[0.95,0.35,0.05]])

 
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)

worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)





#across_generations:-0G--------------------------------------------------------

df1 = filter_dataframe(df, ["is_NG","generation","label_id"], [1,3,[4]])

df1 = filter_dataframe(df, ["is_0G","generation","label_id"], [1,1,[4]])    
df2 = filter_dataframe(df, ["is_0G","generation","label_id"], [1,2,[4]])    
df3 = filter_dataframe(df, ["is_0G","generation","label_id"], [1,3,[4]])    

colors = np.array([[0.05,0.15,0.95],[0.95,0.35,0.05]])
df_list = [df1,df2] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=False)
df_list = [df2,df3] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=False)
df_list = [df1,df3] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=False)
df_list = [df1,df2] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=True)
df_list = [df2,df3] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=True)
df_list = [df1,df3] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=True)


df1 = filter_dataframe(df, ["is_0G","generation","label_id"], [1,1,[5]])    
df2 = filter_dataframe(df, ["is_0G","generation","label_id"], [1,2,[5]])    
df3 = filter_dataframe(df, ["is_0G","generation","label_id"], [1,3,[5]])    

df_list = [df1,df2]
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=False)
df_list = [df2,df3]
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=False)
df_list = [df1,df3]
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=False)

df_list = [df1,df2] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=True)
df_list = [df2,df3] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=True)
df_list = [df1,df3] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=True)




#across_generations:-NG--------------------------------------------------------

dfX = pd.concat((df1,df2,df3))
s=dfX.loc[:,['percent_0.6', 'percent_0.65','percent_0.7','percent_0.75']]
dfX["width_average"] = s.mean(axis=1)
data  = df_to_grouped_array(dfX,"generation","percent_0.5")

colors = np.array([[0.95,0.35,0.05],[0.65,0.1,0.05],[0.5,0.05,0.05]])
plot_grouped_values(data, ["1",'2','3'],figsize=[3.5,6],colors=colors)
