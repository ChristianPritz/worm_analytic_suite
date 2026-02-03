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
import cv2,copy
from measurements import *
from worm_plotter import worm_width_plot, df_to_grouped_array,plot_grouped_values,class_histogram
from classifiers import  classify,run_kmeans_and_show,label_coco_areas,coco_areas_calculation,train_classifier
from annotation_tool_v8 import AnnotationTool
from worm_plotter import *

##---------------------------------------------------------------------------##
#
#                               S E T T I N G S
#
# Global settings for the analysis 
##---------------------------------------------------------------------------##
# global settings
settings = {"rim_score_cutoff":0.05, #cut-off for worms on the image edge
            "debug":False,           #displaying the control plots 
            "use_classifier":False,  #use classifier to eliminate the worm fragments
            "pixel_size":1.4571,        #global pixel size value
            "imsize":3120,
            "AWB":[True,0.05],
            "max_workers": 10}   #maxium a mount of workers in multithreading       


# settings for midline detection 
settings["midline_sttngs"] = {"n_poly":400, # the size of the worm shape polygone
                  "bb_size":100, # outputsize of the backbone
                  "show_plots":False, # show all the plots from parameter search
                  "skip_phase2":True, #skips the fine tuning if heads are a match.
    "coarse_params":  [{"num_start_points": 8}, # number of searches in phase 1
                        {"num_start_points": 6, "search_space": int(400/8)}
                        ],
    "branch_params": [{"num_start_points": 5, "end_points_each": 5, "search_space": 15},# number of searches in phase2
                     {"num_start_points": 5, "end_points_each": 5, "search_space": 3}
                     ],
    "metric_keys":["length",
                   "mean_smoothness",
                   "peak_bend",
                   "end_curvature_ratio",
                   "inside_fraction"
                   ],
    "weights":  {"length": 1.2,
                 "mean_smoothness": -1.1,
                 "peak_bend": 0.0,
                 "end_curvature_ratio": 0,
                 "inside_fraction": 1.0
                 },
    "end_smoothing":{"frac":0.08,        # fraction at each end
                     "window":51,        # strong smoothing window
                     "polyorder":4
                     }
    
    }

base_dir =  '/Users/flaviagreco/Desktop/microscopy/worm_profiler_STARV_HEAT/'
image_dir = '/Users/flaviagreco/Desktop/microscopy/worm_profiler_STARV_HEAT/images' # this is where the images are sitting
point_csv = '/Users/flaviagreco/Desktop/microscopy/worm_profiler_STARV_HEAT/output/annotations_points.csv'
area_json = '/Users/flaviagreco/Desktop/microscopy/worm_profiler_STARV_HEAT/output/annotations_areas.json'
color_csv = '/Users/flaviagreco/Documents/tools/worm_analytic_suite/class_colors.csv'
output_dir= '/Users/flaviagreco/Desktop/microscopy/worm_profiler_STARV_HEAT/output/'
plot_path = '/Users/flaviagreco/Desktop/microscopy/worm_profiler_STARV_HEAT/output/starv_heat_plots'
save_path = '/Users/flaviagreco/Desktop/microscopy/worm_profiler_STARV_HEAT/output/df.csv'
os.makedirs(plot_path,exist_ok=True)


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
# before running this piece of code make sure to have all the images and label files on a single folder. This code line will sort them into their respective folders.
#organize_dataset(base_dir)




# ##---------------------------------------------------------------------------##
# ## this joins the labels and throws out bad labels. 


collect_annotations(settings,
     anno_dir='/Users/flaviagreco/Desktop/microscopy/worm_profiler_STARV_HEAT/labels', # this is where there raw annoations go
     image_dir='/Users/flaviagreco/Desktop/microscopy/worm_profiler_STARV_HEAT/images', # this is where the images are sitting
     color_csv='/Users/flaviagreco/Documents/tools/worm_analytic_suite/class_colors.csv', 
     output_dir='/Users/flaviagreco/Desktop/microscopy/worm_profiler_STARV_HEAT//output' #This is where the joined labels go 
 )


##---------------------------------------------------------------------------##
#              C H E CK    T H E    A N N O T A T I O N S
#
# This allows to change the annotations of animals and the shapes
#
##---------------------------------------------------------------------------##


## Christian's home directories:
#image_dir = '/home/christian/Documents/data/space worms/images' # this is where the images are sitting
#point_csv =  '/home/christian/Documents/data/space worms/output/annotations_points.csv'
#area_json =  '/home/christian/Documents/data/space worms/output/annotations_areas.json'
#color_csv =  '/home/christian/models/worm_analytic_suite/class_colors.csv'
#output_dir= '/home/christian/Documents/data/space worms/images/output/'
#plot_path = '/home/christian/Documents/data/space worms/images/output/plots'
#os.makedirs(plot_path,exist_ok=True)


#only run the line below if you have to change the labels
#tool = AnnotationTool(image_dir,point_csv,area_json,color_csv)


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
    [0.367, 0.697, 0.616],  # R
    [0.604, 0.600, 0.524],  # G
    [0.010, 0.325, 0.945]   # B
])

color_matrix = color_matrix.transpose()


df = analyze_staining_from_json(area_json,
    image_dir,color_matrix,settings,
    output_dir = output_dir,
    df=df)



##---------------------------------------------------------------------------##
#            A S S I G N I N G   C O N D I T I O N   L A B E L S 
##---------------------------------------------------------------------------##
df = pd.read_csv("/Users/flaviagreco/Desktop/microscopy/worm_profiler_STARV_HEAT/output/df.csv")



df = assign_conditions(df,col_name ="image_name_y",naming_func=check_cond_controls,bin_size=20)
df = create_group_labels_controls(df)

#-----------------------------------------------------------------------------#
#
#' SAVE/LOAD  THE DATAFRAME
# 
#-----------------------------------------------------------------------------#

#df.to_csv(save_path)
df = pd.read_csv(save_path)
cols = ['percent_0.05', 'percent_0.1', 'percent_0.15', 'percent_0.2',
       'percent_0.25', 'percent_0.3', 'percent_0.35', 'percent_0.4',
       'percent_0.45', 'percent_0.5', 'percent_0.55', 'percent_0.6',
       'percent_0.65', 'percent_0.7', 'percent_0.75', 'percent_0.8',
       'percent_0.85', 'percent_0.9', 'percent_0.95',]

percents = np.arange(0.05,1,0.05)



#-----------------------------------------------------------------------------#
# Applying pixel size to calibrate the measurements
#-----------------------------------------------------------------------------#
df_backup = copy.deepcopy(df)
for i in cols :
    df[i] = df[i]/settings["pixel_size"] 
df["length"] = df["length"]/settings["pixel_size"]
df["area"] = df["area"]/settings["pixel_size"]**2





##---------------------------------------------------------------------------##
#                 Plotting worm with graphs.... 
##---------------------------------------------------------------------------##
group_colors = np.array([[0.95,0.24,0.08],[0.15,0.45,0.95],[0.6,0.6,0.6]])

#0g - LH - NG)
# worm width plot------------------------------------------------------------##
df_starve_con = filter_dataframe(df, ["is_starved","is_control","label_id"], [1,1,[4]])
df_starve_treated = filter_dataframe(df, ["is_starved","is_treated","label_id"], [1,1,[4]])
df_hs_con = filter_dataframe(df, ["is_heatshock","is_control","label_id"], [1,1,[4]])
df_hs_treated = filter_dataframe(df, ["is_heatshock","is_treated","label_id"], [1,1,[4]])



#0G 
df_list = [df_starve_con,df_starve_con]
flips = [False,True]
colors = group_colors[[0,0],:]
axObj = worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_starve_con",plot_path)

#LH 
df_list = [df_starve_treated,df_starve_treated]
flips = [False,True]
colors = group_colors[[1,1],:]
axObj = worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_starve_treated",plot_path)


#overall comparisons 
df_list = [df_starve_con,df_starve_treated]
colors = group_colors[[2,0],:]
flips = [False,True]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_starved_scaled",plot_path)
flips = [False,False]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)
save_plot(axObj[0],"worm_width_starved",plot_path)

#HEATSHOCK







# metric plots ----------------------------------------------------------------
#
#------------------------------------------------------------------------------




#df_adult = df[df["label_id"]==4]
s=df.loc[:,['percent_0.5','percent_0.55','percent_0.6', 'percent_0.65','percent_0.7','percent_0.75']]
df["width_average"] = s.mean(axis=1)

group_colors = np.array([[0.95,0.24,0.08],[0.15,0.45,0.95],[0.6,0.6,0.6],[0.6,0.6,0.6]])
metrics = ["area","length","intensity","width_average"]
cond_labels = ["is_con","is_treated"]
conds = ["control","treated"]
props = [{"xlim":"AUTO","ylim":"AUTO","xlabel":'',"ylabel":'Area (µm2)'}, #area
         {"xlim":"AUTO","ylim":"AUTO","xlabel":'',"ylabel":'Length (µm)'}, #length
         {"xlim":"AUTO","ylim":"AUTO","xlabel":'',"ylabel":'Intensity (a.u.)'}, #intensity
         {"ylim":[30,120],"xlim":"AUTO","xlabel":'',"ylabel":'Width (µm)'},]#width_average




df_starve = filter_dataframe(df, ["is_starved","label_id"], [1,[4]])
for mDx,i in enumerate(metrics): 
    prop = props[mDx]
    #df_adult = df[df["label_id"]=4]
    data,grps  = df_to_grouped_array(df_starve,"group_identifier",i)
    axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,plot_props=prop)
    name = i + "_STARVED_comparison"
    save_plot(axObj[0],name,plot_path)
    if i == "intensity":
        axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,logY=True,plot_props=prop)
        name = i + "_STARVED_comparison_logY"
        save_plot(axObj[0],name,plot_path)
    
df_hs = filter_dataframe(df, ["is_heatshock","label_id"], [1,[4]])
for mDx,i in enumerate(metrics): 
    prop = props[mDx]
    #df_adult = df[df["label_id"]=4]
    data,grps  = df_to_grouped_array(df_hs,"group_identifier",i)
    axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,plot_props=prop)
    name = i + "_heatshock_comparison"
    save_plot(axObj[0],name,plot_path)
    if i == "intensity":
        axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,logY=True,plot_props=prop)
        name = i + "_heatshock_comparison_logY"
        save_plot(axObj[0],name,plot_path)

    




















