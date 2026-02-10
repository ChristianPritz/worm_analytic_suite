#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:22:07 2026

@author: wormulon
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
import json


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






comparison_values = {"larval_stages":[],"larval_distribution":np.nan}



## werk: 
image_dir = '/media/my_device/space worms/worm_profiler_parental/images' # this is where the images are sitting
point_csv = '/media/my_device/space worms/worm_profiler_parental/output/annotations_points.csv'
area_json = '/media/my_device/space worms/worm_profiler_parental/output/annotations_areas.json'
color_csv = '/home/wormulon/models/worm_analytic_suite/class_colors.csv'
output_dir= '/media/my_device/space worms/worm_profiler_parental/images/output/'
plot_path = '/media/my_device/space worms/worm_profiler_parental/images/plots'
save_path = '/media/my_device/space worms/worm_profiler_parental/output/df_parental.csv'
p0_path = '/media/my_device/space worms/worm_profiler_parental/output/p0_metrics.json'



os.makedirs(plot_path,exist_ok=True)












##---------------------------------------------------------------------------##
#            A S S I G N I N G   C O N D I T I O N   L A B E L S 
##---------------------------------------------------------------------------##

df = pd.read_csv(save_path)
df = assign_conditions(df,col_name ="image_name_y",naming_func=check_cond,bin_size=20)
df = create_group_labels(df)


## putting the measurements  backinto  µm -----------------------------------##
#
for i in cols :
    df[i] = df[i]/settings["pixel_size"] 
df["length"] = df["length"]/settings["pixel_size"]
df["area"] = df["area"]/settings["pixel_size"]**2
df["is_real_world_units"] = True




group_colors = np.array([[0.95,0.24,0.08],[0.15,0.45,0.95],[0.6,0.6,0.6],[0.6,0.6,0.6]])
#outline graph 
df_list = [df_adult,df_adult]
flips = [False,True]
colors = group_colors[[0,0],:]
axObj = worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_P0_con",plot_path)





# metric plots ----------------------------------------------------------------
#
#------------------------------------------------------------------------------




#df_adult = df[df["label_id"]==4]
s=df.loc[:,['percent_0.5','percent_0.55','percent_0.6', 'percent_0.65','percent_0.7','percent_0.75']]
df["width_average"] = s.mean(axis=1)

df_adult = filter_dataframe(df, ["label_id"], [[4]])

group_colors = np.array([[0.95,0.24,0.08],[0.15,0.45,0.95],[0.6,0.6,0.6],[0.6,0.6,0.6]])
metrics = ["area","length","intensity","width_average"]
cond_labels = ["is_con","is_treated"]
conds = ["control","treated"]
props = [{"xlim":"AUTO","ylim":"AUTO","xlabel":'',"ylabel":'Area (µm2)'}, #area
         {"xlim":"AUTO","ylim":"AUTO","xlabel":'',"ylabel":'Length (µm)'}, #length
         {"xlim":"AUTO","ylim":"AUTO","xlabel":'',"ylabel":'Intensity (a.u.)'}, #intensity
         {"ylim":[30,120],"xlim":"AUTO","xlabel":'',"ylabel":'Width (µm)'},]#width_average


for mDx,i in enumerate(metrics): 
    prop = props[mDx]
    #df_adult = df[df["label_id"]=4]
    data,grps  = df_to_grouped_array(df_adult,"group_identifier",i)
    axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,plot_props=prop)
    name = i + "_P0"
    save_plot(axObj[0],name,plot_path)
    if i == "intensity":
        axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,logY=True,plot_props=prop)
        name = i + "_P0_comparison_logY"
        save_plot(axObj[0],name,plot_path)
    m = [np.nanmean(data[:,0],axis=0),np.nanstd(data[:,0],axis=0)]
    print(i,m)
    comparison_values[i] = m
    
comparison_values["P0_larval_labels"]=df.label_id.to_list()



with open(p0_path, "w") as f:
    json.dump(comparison_values, f)

    

l_path = "/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/output/plots/"

fig,ax = load_plot("area_0G_over_generations_single_condition",l_path)

add_mean_std_lines(fig, ax, 120000, 5000,["area_0G_over_generations_single_condition",l_path])



name1 = "life_stage_histogram_NGoverall"
name2 = "life_stage_histogram_0Goverall"

merge_pickled_barplots_over_ax1(name1, name2, l_path)


