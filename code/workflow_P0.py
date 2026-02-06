#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:22:07 2026

@author: wormulon
"""

import numpy as np


comparison_values = {
               "larval_stages":[],
               "larval_distribution":np.nan,
               },


df_adult = filter_dataframe(df, ["label_id"], [[4]])



group_colors = np.array([[0.95,0.24,0.08],[0.15,0.45,0.95],[0.6,0.6,0.6],[0.6,0.6,0.6]])
#outline graph 
df_list = [df_adult,df_adult]
flips = [False,True]
colors = group_colors[[0,0],:]
axObj = worm_width_plot(df_list, cols, percents,'length', figsize=(6,1),colors=colors,flip=flips,w_ratio=1.5,scale_by_length=True)
save_plot(axObj[0],"worm_width_starve_con",plot_path)





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

for mDx,i in enumerate(metrics): 
    prop = props[mDx]
    #df_adult = df[df["label_id"]=4]
    data,grps  = df_to_grouped_array(df_adult,"group_identifier",i)
    axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,plot_props=prop)
    name = i + "_P0"
    save_plot(axObj[0],name,plot_path)
    if i == "intensity":
        axObj = plot_grouped_values(data, grps,figsize=[3.5,6],colors=group_colors,logY=True,plot_props=prop)
        name = i + "_STARVED_comparison_logY"
        save_plot(axObj[0],name,plot_path)
    m = [np.nanmean(data[0],axis=0),np.nanstd(data[0],axis=0)]
    comparison_values[i] = m
    
comparison_values["P0_larval_labels"]=df.label_id



    

l_path = "/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images/output/plots/"

fig,ax = load_plot("area_0G_over_generations_single_condition",l_path)

add_mean_std_lines(fig, ax, 120000, 5000,["area_0G_over_generations_single_condition",l_path])



name1 = "life_stage_histogram_NGoverall"
name2 = "life_stage_histogram_0Goverall"

merge_pickled_barplots_over_ax1(name1, name2, l_path)


