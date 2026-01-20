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
from measurements import collect_annotations, analyze_annotations,analyze_thickness,analyze_staining_from_json, assign_conditions,organize_dataset, create_group_labels
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
            "max_workers": 16}   #maxium a mount of workers in multithreading       


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


## home:
image_dir = '/home/christian/Documents/data/space worms/images' # this is where the images are sitting
point_csv =  '/home/christian/Documents/data/space worms/output/annotations_points.csv'
area_json =  '/home/christian/Documents/data/space worms/output/annotations_areas.json'
color_csv =  '/home/christian/models/worm_analytic_suite/class_colors.csv'
output_dir= '/home/christian/Documents/data/space worms/images/output/'
plot_path = '/home/christian/Documents/data/space worms/images/output/plots'
save_path = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/output/df.csv'
os.makedirs(plot_path,exist_ok=True)


## werk: 
image_dir = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images' # this is where the images are sitting
point_csv = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_points.csv'
area_json = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_areas.json'
color_csv = '/home/wormulon/models/worm_analytic_suite/class_colors.csv'
output_dir= '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/output/'
plot_path = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/output/plots'
save_path = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/test.csv'

os.makedirs(plot_path,exist_ok=True)

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


df = analyze_staining_from_json(area_json,
    image_dir,color_matrix,settings,
    output_dir = output_dir,
    df=df)

##---------------------------------------------------------------------------##
#            A S S I G N I N G   C O N D I T I O N   L A B E L S 
##---------------------------------------------------------------------------##

df = assign_conditions(df,col_name ="image_name_y")
df = create_group_labels(df)





#-----------------------------------------------------------------------------#
# Applying pixel size to calibrate the measurements
#-----------------------------------------------------------------------------#
df_backup = copy.deepcopy(df)
for i in cols :
    df[i] = df[i]/settings["pixel_size"] 
df["length"] = df["length"]/settings["pixel_size"]
df["area"] = df["area"]/settings["pixel_size"]**2


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


##---------------------------------------------------------------------------##
#                 Plotting worm with graphs.... 
##---------------------------------------------------------------------------##
group_colors = np.array([[0.95,0.24,0.08],[0.15,0.45,0.95],[0.6,0.6,0.6]])

#0g - LH - NG)
# worm width plot------------------------------------------------------------##
df_NG = filter_dataframe(df, ["is_NG","generation","label_id"], [1,[1,2,3],[4]])
df_0G = filter_dataframe(df, ["is_0G","generation","label_id"], [1,[1,2,3],[4]])
df_LH = filter_dataframe(df, ["is_LH","generation","label_id"], [1,[1,2,3],[4]])
df_NG1 = filter_dataframe(df_NG, ["generation"], [1,])
df_NG2 = filter_dataframe(df_NG, ["generation"], [2,])
df_NG3 = filter_dataframe(df_NG, ["generation"], [3,])
df_0G1 = filter_dataframe(df_0G, ["generation"], [1,])
df_0G2 = filter_dataframe(df_0G, ["generation"], [2,])
df_0G3 = filter_dataframe(df_0G, ["generation"], [3,])
df_LH1 = filter_dataframe(df_LH, ["generation"], [1,])
df_LH2 = filter_dataframe(df_LH, ["generation"], [2,])
df_LH3 = filter_dataframe(df_LH, ["generation"], [3,])
#single conditions
#0G 
df_list = [df_0G,df_0G]
flips = [False,True]
colors = group_colors[[0,0],:]
axObj = worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_overall_0G",plot_path)

#LH 
df_list = [df_LH,df_LH]
flips = [False,True]
colors = group_colors[[1,1],:]
axObj = worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_overall_LH",plot_path)

#NG 
df_list = [df_NG,df_NG]
flips = [False,True]
colors = group_colors[[2,2],:]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_overall_NG",plot_path)

#overall comparisons 
df_list = [df_NG,df_0G]
colors = group_colors[[2,0],:]
flips = [False,True]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_NG_vs_0G_scaled",plot_path)
flips = [False,False]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)
save_plot(axObj[0],"worm_width_NG_vs_0G",plot_path)


df_list = [df_NG,df_LH]
colors = group_colors[[2,1],:]
flips = [False,True]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_NG_vs_LH_scaled",plot_path)
flips = [False,False]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)
save_plot(axObj[0],"worm_width_NG_vs_LH",plot_path)


df_list = [df_0G,df_LH]
colors = group_colors[[2,1],:]
flips = [False,True]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_0G_vs_LH_scaled",plot_path)
flips = [False,False]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)
save_plot(axObj[0],"worm_width_0G_vs_LH",plot_path)



# across generations
#0G
df_list = [df_0G1,df_0G2]
colors = np.tile(np.array([[0.5,0.75]]).T,(1,3)) * group_colors[[0,0],:]
flips = [False,True]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_generations_0G_1_2_scaled",plot_path)
flips = [False,False]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)
save_plot(axObj[0],"worm_width_generations_0G_1_2",plot_path)

df_list = [df_0G2,df_0G3]
colors = np.tile(np.array([[0.75,1]]).T,(1,3)) * group_colors[[0,0],:]
flips = [False,True]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_generations_0G_2_3_scaled",plot_path)
flips = [False,False]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)
save_plot(axObj[0],"worm_width_generations_0G_2_3",plot_path)

#LH
df_list = [df_LH1,df_LH2]
colors = np.tile(np.array([[0.5,0.75]]).T,(1,3)) * group_colors[[1,1],:]
flips = [False,True]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
flips = [False,False]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)
df_list = [df_LH2,df_LH3]
colors = np.tile(np.array([[0.75,1]]).T,(1,3)) * group_colors[[1,1],:]
flips = [False,True]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
flips = [False,False]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)


#NG
df_list = [df_NG1,df_NG2]
colors = np.tile(np.array([[0.5,0.75]]).T,(1,3)) * group_colors[[2,2],:]
flips = [False,True]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
flips = [False,False]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)
df_list = [df_NG2,df_NG3]
colors = np.tile(np.array([[0.75,1]]).T,(1,3)) * group_colors[[2,2],:]
flips = [False,True]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
flips = [False,False]
axObj =worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=False)



#consistency test: (checking if group order is maintained (debug))
    
a = np.random.random((10, 1)) + 1
b = np.random.random((10, 1)) + 11
c = np.random.random((10, 1)) + 21

l1 = np.full((10, 1), 'a')
l2 = np.full((10, 1), 'b')
l3 = np.full((10, 1), 'c')

data = np.vstack((a, b, c)).ravel()
labels = np.vstack((l1, l2, l3)).ravel()

test_df = pd.DataFrame({
    "data": data,
    "labels": labels
})
data,grps  = df_to_grouped_array(test_df,"labels","data")
colors = np.array([[0.15,0.45,0.95],[0.15,0.45,0.95],[0.15,0.45,0.95]])
axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=colors)





# metric plots ----------------------------------------------------------------
#
#------------------------------------------------------------------------------


#df_adult = df[df["label_id"]==4]
s=df.loc[:,['percent_0.5','percent_0.55','percent_0.6', 'percent_0.65','percent_0.7','percent_0.75']]
df["width_average"] = s.mean(axis=1)

group_colors = np.array([[0.95,0.24,0.08],[0.15,0.45,0.95],[0.6,0.6,0.6]])
metrics = ["area","length","intensity","width_average"]
cond_labels = ["is_0G","is_LH","is_NG"]
conds = ["0G","LH","NG"]
props = [{"xlim":"AUTO","ylim":"AUTO","xlabel":'',"ylabel":'Area (µm2)'}, #area
         {"xlim":"AUTO","ylim":"AUTO","xlabel":'',"ylabel":'Length (µm)'}, #length
         {"xlim":"AUTO","ylim":"AUTO","xlabel":'',"ylabel":'Intensity (a.u.)'}, #intensity
         {"ylim":[80,160],"xlim":"AUTO","xlabel":'',"ylabel":'Width (µm)'},]#width_average

for mDx,i in enumerate(metrics): 
    prop = props[mDx]
    df_adult = df[df["label_id"]<6]
    #df_adult = df[df["label_id"]=4]
    data,grps  = df_to_grouped_array(df_adult,"group_identifier",i)
    axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,plot_props=prop)
    name = i + "_overall_comparison"
    save_plot(axObj[0],name,plot_path)
    if i == "intensity":
        axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,logY=True,plot_props=prop)
        name = i + "_overall_comparison_logY"
        save_plot(axObj[0],name,plot_path)
    
    # COMPARISONS BETWEEN CONDITIONS
    # by generations
    for idx in range(1,4):
        df_cond = df_adult[df_adult["generation"]==idx]
        data,grps  = df_to_grouped_array(df_cond,"group_identifier",i)
        axObj = plot_grouped_values(data,grps,figsize=[3.5,6],colors=group_colors,plot_props=prop)
        name = i + "_comparison_between_groups_for_generation_" + str(idx) 
        print(name)
        save_plot(axObj[0],name,plot_path)
        if i == "intensity":
            axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,logY=True,plot_props=prop)
            name = name + "_logY"
            save_plot(axObj[0],name,plot_path)
        
    # by trials
    for idx in range(1,4):
        df_cond = df_adult[df_adult["trial"]==idx]
        data,grps  = df_to_grouped_array(df_cond,"group_identifier",i)
        axObj = plot_grouped_values(data,grps,figsize=[3.5,6],colors=group_colors,plot_props=prop)
        name = i + "_comparison_between_groups_for_trial_" + str(idx) 
        print(name)
        save_plot(axObj[0],name,plot_path)
        if i == "intensity":
            axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,logY=True,plot_props=prop)
            name = name + "_logY"
            save_plot(axObj[0],name,plot_path)
    
    
    # SINGLE CONDTIONS
    # by generations 

    for idx,cont in enumerate(zip(conds,cond_labels)):
        label = cont[0]
        selector = cont[1]
        df_cond = df_adult[df_adult[selector]==1]
        data,grps  = df_to_grouped_array(df_cond,"generation",i)
        colors = np.tile(group_colors[idx,:],(3,1))
        axObj = plot_grouped_values(data,grps,figsize=[3.5,6],colors=colors,plot_props=prop)
        name = i + "_" + label + "_over_generations_single_condition" 
        print(name)
        save_plot(axObj[0],name,plot_path)
        if i == "intensity":
            axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=colors,logY=True,plot_props=prop)
            name = name + "_logY"
            save_plot(axObj[0],name,plot_path)
     
    # by trials
    for idx,cont in enumerate(zip(conds,cond_labels)):
        label = cont[0]
        selector = cont[1]
        df_cond = df_adult[df_adult[selector]==1]
        data,grps  = df_to_grouped_array(df_cond,"trial",i)
        colors = np.tile(group_colors[idx,:],(3,1))
        axObj = plot_grouped_values(data,grps,figsize=[3.5,6],colors=colors,plot_props=prop)
        name = i + "_" + label + "_over_generations_single_condition" 
        print(name)
        save_plot(axObj[0],name,plot_path)
        if i == "intensity":
            axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=colors,logY=True,plot_props=prop)
            name = name + "_logY"
            save_plot(axObj[0],name,plot_path)


# Larval and life stage distributions -----------------------------------------
#
#------------------------------------------------------------------------------

#loading the class labels
clss = pd.read_csv(color_csv,header=None)
cond_labels = ["is_0G","is_LH","is_NG"]
conds = ["0G","LH","NG"]
group_colors = np.array([[0.95,0.24,0.08],[0.15,0.45,0.95],[0.6,0.6,0.6]])
display_classes =       clss.iloc[3:13,:]

disp_clss = display_classes
disp_clss = disp_clss.drop(6)



for idx, container in enumerate(zip(conds,cond_labels)):
    label = container[0]
    selector = container[1]
    #overall
    df_cond = df[df[selector]==1]
    data = df_cond["label_id"]
    axObj = class_histogram(data,clss,show_counts=True,color = group_colors[idx,:],normalize=True,ax_labels=disp_clss)
    name = "life_stage_histogram_" + label + "overall"
    save_plot(axObj[0],name,plot_path)


    #Across generations
    for i in range(1,4):
        df_cond_x = df_cond[df_cond["generation"]==i]
        data = df_cond_x["label_id"]
        name = "life_stages_histogram_" + label + "_generation_" + str(i+1)
        axObj = class_histogram(data,clss,show_counts=True,color = group_colors[idx,:],normalize=True,ax_labels=disp_clss)
        save_plot(axObj[0],name,plot_path)
    
    #Across trials    
    for i in range(1,4):
        df_cond_x = df_cond[df_cond["trial"]==i]
        data = df_cond_x["label_id"]
        name = "life_stages_histogram_" + label + "_trial_" + str(i+1)
        axObj = class_histogram(data,clss,show_counts=True,color = group_colors[idx,:],normalize=True,ax_labels=disp_clss)  
        save_plot(axObj[0],name,plot_path)



















