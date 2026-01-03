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
from measurements import collect_annotations, analyze_annotations,analyze_thickness,analyze_staining_from_json, assign_conditions,organize_dataset, create_group_labels
from worm_plotter import worm_width_plot, df_to_grouped_array,plot_grouped_values,class_histogram
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


# ## this copies all the files together into a single image directory 
# ## and a single labels directory (output)

# organize_dataset("/media/my_device/space worms/makesenseai_analyzed_images/to analyze")



# ##---------------------------------------------------------------------------##
# ## this joins the labels and throws out bad labels. 

# settings = {"rim_score_cutoff":0.05,"debug":False,"use_classifier":True}
# collect_annotations(settings,
#     anno_dir='/media/my_device/space worms/makesenseai_analyzed_images/to analyze/labels', # this is where there raw annoations go
#     image_dir='/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images', # this is where the images are sitting
#     color_csv='/home/wormulon/models/worm_analytic_suite/class_colors.csv', 
#     output_dir='/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output' #This is where the joined labels go 
# )


##---------------------------------------------------------------------------##
#              C H E CK    T H E    A N N O T A T I O N S
#
# This allows to change the annotations of animals and the shapes
#
##---------------------------------------------------------------------------##

image_dir = '/home/christian/Documents/data/space worms/images' # this is where the images are sitting
point_csv =  '/home/christian/Documents/data/space worms/output/annotations_points.csv'
area_json =  '/home/christian/Documents/data/space worms/output/annotations_areas.json'
color_csv =  '/home/christian/models/worm_analytic_suite/class_colors.csv'

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
settings = {"rim_score_cutoff":0.05,"debug":False,"use_classifier":True}
df = analyze_annotations(
    areas_json=area_json,
    points_csv=point_csv,
    image_dir=image_dir,
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
            "imsize":3120,
            "AWB":[True,0.05],
            "debug":False}
            

df = analyze_staining_from_json(area_json,
    image_dir,color_matrix,settings,
    output_dir = '/home/christian/Documents/data/space worms/images/output/',
    df=df)

##---------------------------------------------------------------------------##
#            A S S I G N I N G   C O N D I T I O N   L A B E L S 
##---------------------------------------------------------------------------##

df = assign_conditions(df,col_name ="image_name_y")
df = create_group_labels(df)

##---------------------------------------------------------------------------##
#                 Plotting
##---------------------------------------------------------------------------##

cols = ['percent_0.05', 'percent_0.1', 'percent_0.15', 'percent_0.2',
       'percent_0.25', 'percent_0.3', 'percent_0.35', 'percent_0.4',
       'percent_0.45', 'percent_0.5', 'percent_0.55', 'percent_0.6',
       'percent_0.65', 'percent_0.7', 'percent_0.75', 'percent_0.8',
       'percent_0.85', 'percent_0.9', 'percent_0.95',]

percents = np.arange(0.05,1,0.05)


from worm_plotter import worm_width_plot, filter_dataframe,df_to_grouped_array


# worm width plot------------------------------------------------------------##
df4 = filter_dataframe(df, ["is_NG","generation","label_id"], [1,3,[4]])
df5 = filter_dataframe(df, ["is_0G","generation","label_id"], [1,3,[4]])

df_list = [df4,df5]

flips = [False,False]
colors = np.array([[0.05,0.15,0.95],[0.95,0.35,0.05]])
 
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)



#across_generations:-0G--------------------------------------------------------

# sorting the data into generations 
df1 = filter_dataframe(df, ["is_0G","generation","label_id"], [1,1,[4]])    
df2 = filter_dataframe(df, ["is_0G","generation","label_id"], [1,2,[4]])    
df3 = filter_dataframe(df, ["is_0G","generation","label_id"], [1,3,[4]])    

colors = np.array([[0.05,0.15,0.95],[0.95,0.35,0.05]])

#generation 1 vs 2
df_list = [df1,df2] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=False)
#generation 2 vs 3
df_list = [df2,df3] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=False)
#generation 1 vs 3
df_list = [df1,df3] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=False)


df_list = [df1,df2] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=True)
df_list = [df2,df3] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=True)
df_list = [df1,df3] #blue, #red
worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=3,scale_by_length=True)


#STAINING INTENSITY across_generations:-NG--------------------------------------------------------
group_colors = np.array([[0.95,0.24,0.08],[0.15,0.45,0.95],[0.6,0.6,0.6]])
df_adult = df[df["label_id"]==4]

data,grps  = df_to_grouped_array(df_adult,"group_identifier","intensity")
print(grps)

plot_grouped_values(data,grps,figsize=[3.5,6],colors=group_colors,logY=True)

df_LH = df_adult[df_adult["is_LH"]==1]
data,grps  = df_to_grouped_array(df_LH,"generation","intensity")
colors = np.array([[0.15,0.45,0.95],[0.15,0.45,0.95],[0.15,0.45,0.95]])
plot_grouped_values(data, grps,figsize=[3.5,6],colors=colors,logY=True)


df_0G = df_adult[df_adult["is_0G"]==1]
data,grps  = df_to_grouped_array(df_0G,"generation","intensity")
colors = np.array([[0.95,0.24,0.08],[0.95,0.24,0.08],[0.95,0.24,0.08]])
plot_grouped_values(data, grps,figsize=[3.5,6],colors=colors,logY=True)


df_NG = df_adult[df_adult["is_NG"]==1]
data,grps  = df_to_grouped_array(df_NG,"generation","intensity")
colors = np.array([[0.6,0.6,0.6],[0.6,0.6,0.6],[0.6,0.6,0.6]])
plot_grouped_values(data,grps,figsize=[3.5,6],colors=colors,logY=True)


# WIDTH -----------------------------------------------------------------------
#
#------------------------------------------------------------------------------
df_adult = df[df["label_id"]<6]
s=df_adult.loc[:,['percent_0.5','percent_0.55','percent_0.6', 'percent_0.65','percent_0.7','percent_0.75']]
df_adult["width_average"] = s.mean(axis=1)
data,grps  = df_to_grouped_array(df_adult,"group_identifier","width_average")
plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors)

# across_generations:-LG-------------------------------------------------------
df_LH = df_adult[df_adult["is_NG"]==1]
data,grps  = df_to_grouped_array(df_LH,"generation","width_average")
colors = np.array([[0.15,0.45,0.95],[0.15,0.45,0.95],[0.15,0.45,0.95]])
plot_grouped_values(data, grps,figsize=[3.5,6],colors=colors)

# across_generations:-0G-------------------------------------------------------
df_0G = df_adult[df_adult["is_0G"]==1]
data,grps  = df_to_grouped_array(df_0G,"generation","width_average")
colors = np.array([[0.95,0.24,0.08],[0.95,0.24,0.08],[0.95,0.24,0.08]])
plot_grouped_values(data, grps,figsize=[3.5,6],colors=colors)

# across_generations:-0G-------------------------------------------------------
df_NG = df_adult[df_adult["is_NG"]==1]
data,grps  = df_to_grouped_array(df_NG,"generation","width_average")
colors = np.array([[0.6,0.6,0.6],[0.6,0.6,0.6],[0.6,0.6,0.6]])
plot_grouped_values(data, grps,figsize=[3.5,6],colors=colors)



# AREA---------------------------------------------------------------------
#
#------------------------------------------------------------------------------
df_adult = df[df["label_id"]==4]

data,grps  = df_to_grouped_array(df_adult,"group_identifier","area")
plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors)

df_LH = df_adult[df_adult["is_LH"]==1]
data,grps  = df_to_grouped_array(df_LH,"generation","area")
colors = np.array([[0.15,0.45,0.95],[0.15,0.45,0.95],[0.15,0.45,0.95]])
plot_grouped_values(data,grps,figsize=[3.5,6],colors=colors)


df_0G = df_adult[df_adult["is_0G"]==1]
data,grps  = df_to_grouped_array(df_0G,"generation","area")
colors = np.array([[0.95,0.24,0.08],[0.95,0.24,0.08],[0.95,0.24,0.08]])
plot_grouped_values(data,grps,figsize=[3.5,6],colors=colors)


df_NG = df_adult[df_adult["is_NG"]==1]
data,grps  = df_to_grouped_array(df_NG,"generation","area")
colors = np.array([[0.6,0.6,0.6],[0.6,0.6,0.6],[0.6,0.6,0.6]])
plot_grouped_values(data,grps,figsize=[3.5,6],colors=colors)


# Length-----------------------------------------------------------------------
#
#------------------------------------------------------------------------------
df_adult = df[df["label_id"]<6]


data,grps  = df_to_grouped_array(df_adult,"group_identifier","length")
plot_grouped_values(data,grps,figsize=[3.5,6],colors=group_colors)

df_LH = df_adult[df_adult["is_LH"]==1]
data,grps  = df_to_grouped_array(df_LH,"generation","length")
colors = np.array([[0.15,0.45,0.95],[0.15,0.45,0.95],[0.15,0.45,0.95]])
plot_grouped_values(data, grps,figsize=[3.5,6],colors=colors)


df_0G = df_adult[df_adult["is_0G"]==1]
data,grps  = df_to_grouped_array(df_0G,"generation","length")
colors = np.array([[0.95,0.24,0.08],[0.95,0.24,0.08],[0.95,0.24,0.08]])
plot_grouped_values(data, grps,figsize=[3.5,6],colors=colors)


df_NG = df_adult[df_adult["is_NG"]==1]
data,grps  = df_to_grouped_array(df_NG,"generation","length")
colors = np.array([[0.6,0.6,0.6],[0.6,0.6,0.6],[0.6,0.6,0.6]])
plot_grouped_values(data, grps,figsize=[3.5,6],colors=colors)



# Larval and life stage distributions -----------------------------------------
#
#------------------------------------------------------------------------------

#loading the class labels
color_csv =  '/home/christian/models/worm_analytic_suite/class_colors.csv'
clss = pd.read_csv(color_csv)

df_LH = df[df["is_LH"]==1]
data = df_LH["label_id"]
class_histogram(data,clss,show_counts=True,color = [0.15,0.45,0.95])
for i in range(1,4):
    df_LH_x = df_LH[df_LH["generation"]==i]
    data = df_LH_x["label_id"]
    class_histogram(data,clss,show_counts=True,color = [0.15,0.45,0.95])

df_0G = df[df["is_0G"]==1]
data = df_0G["label_id"]
class_histogram(data,clss,show_counts=True,color = [0.95,0.24,0.08],normalize=True)
for i in range(1,4):
    df_0G_x = df_0G[df_0G["generation"]==i]
    data = df_0G_x["label_id"]
    class_histogram(data,clss,show_counts=True,color = [0.95,0.24,0.08],normalize=True)

df_NG = df[df["is_NG"]==1]
data = df_NG["label_id"]
class_histogram(data,clss,show_counts=True,color = [0.6,0.6,0.6])
for i in range(1,4):
    df_NG_x = df_NG[df_NG["generation"]==i]
    data = df_NG_x["label_id"]
    class_histogram(data,clss,show_counts=True,color = [0.6,0.6,0.6])

