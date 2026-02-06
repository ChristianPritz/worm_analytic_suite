#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:09:51 2025

@author: christian
"""
import pandas as pd
import numpy as np
import os, pickle,scipy
import matplotlib.pyplot as plt
from IPython import embed
from IPython.display import display
from matplotlib.patches import Rectangle

# def df_to_grouped_array(df: pd.DataFrame, sort_col: str, value_col: str):
#     # Group by sort_col and extract lists of value_col
#     groups = df.groupby(sort_col, sort=False)[value_col].apply(list)
    
#     # Capture the order of the unique sort_col values
#     order = groups.index.to_list()
    
#     # Find the maximum group length to pad uneven groups
#     max_len = max(len(lst) for lst in groups)
#     result = np.full((max_len, len(groups)), np.nan)
#     for i, lst in enumerate(groups):
#         result[:len(lst), i] = lst
    
#     return result, order

def df_to_grouped_array(df: pd.DataFrame, sort_col: str, value_col: str):
    print("Carving")
    un_val = np.unique(df[sort_col])
    
    result = np.full((len(df), len(un_val)), np.nan)
    for idx,i in enumerate(un_val):
        vals = df[value_col][df[sort_col] == i]
        result[0:len(vals),idx] = vals;
     
    result = result[np.sum(~np.isnan(result),1)==result.shape[1]]
    return result, un_val 
    
    
    



def filter_dataframe(df, columns, values):

    if len(columns) != len(values):
        raise ValueError("columns and values must have the same length")

    filtered_df = df.copy()
    for col, val in zip(columns, values):
        if isinstance(val,list):
            conds = []
            for j in val:
                conds.append(filtered_df[col] == j)
            is_this = np.logical_or.reduce(conds)
            filtered_df = filtered_df[is_this]
            
        else:
            filtered_df = filtered_df[filtered_df[col] == val]
    
    return filtered_df

def worm_width_plot(df_container, columns, percents, lengths, figsize=(6,1),
                    colors=[[0.05,0.15,0.89]],w_ratio=1,flip = None,
                    scale_by_length=False):
    """
    Plot mean +/- std of dataframe columns against given y-values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing numeric columns.
    columns : list or range
        Columns to analyze (names or index range).
    y_values : array-like
        Y-position for each column (must match number of selected columns).
    figsize : tuple
        Size of the plot.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=600)
    
    for idx,df in enumerate(df_container):
        tail_x = np.array([1,1.05,1.08,1.10,1.12])
        head_x = np.array([0,0.01,0.02,0.03,0.04])
        x = np.hstack((head_x,percents,tail_x))
        # Select columns
        if isinstance(columns, (list, tuple)):
            data = df[columns].to_numpy()
        else:
            data = df.iloc[:, columns].to_numpy()
    
        l_mean = np.nanmean(df[lengths])
        l_sdev = np.nanstd(df[lengths])
    
        
        # Compute statistics
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        head_y = np.array([0,0.4,0.6,0.75,0.86]) * means[0]
       
        
        tail_y = np.array([0.5,0.2,0.125,0.05,0]) * means[-1]
        
        y = np.hstack((head_y,means,tail_y))
        # Plot
        
        if scale_by_length:
            x =  x/1.12  * l_mean
            p = percents/1.12 * l_mean
            reverser = l_mean
        else:
            x =  x/1.12  * 1000
            p = percents/1.12 * 1000
            reverser = 1000
        
        #Scaling width to length for display
        
        y = y * w_ratio
        means = means * w_ratio
        stds = stds * w_ratio
    
        
        #Reversing the order 
    
        x = np.abs(x-reverser) 
        p = np.abs(p-reverser) 
 
        
        #flip below 0 for comparison
        l_delinator = 32
        if flip is not None:
            if flip[idx]:
                y = y * -1
                means = means * -1
                stds = stds * -1
                l_delinator = l_delinator *-1        

        print(x)
        ax.plot(x,y, linewidth=0.5, color=colors[idx,:])
        ax.fill_between(p, means + stds, means - stds, alpha=0.3, color=colors[idx,:],linewidth=0)
        ax.scatter([x[0]-l_sdev,x[0]+l_sdev],[0,0],color=colors[idx,:],marker='+') 
        ax.fill_between([x[0]-l_sdev,x[0]+l_sdev], l_delinator, 0, alpha=0.3, color=colors[idx,:],linewidth=0)
        
    
    ax.set_aspect("equal")

 
    # Hide y-axis completely
    ax.get_yaxis().set_visible(False)
    
    
    # Move x-axis to y = 0
    ax.spines['bottom'].set_position(('data', 0))
    xl = ax.get_xlim()
    ax.set_xlim((0,xl[1]))
    # Hide all other spines except bottom
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Remove x-axis ticks
    ax.xaxis.set_ticks_position('none')
    ax.set_xticks([])
    


    plt.tight_layout()
    plt.show()
    return (fig,ax)


def instantiate_plot_props(plot_props=None):
        # default_props = {
        #     'ylim': [], # do not use 'AUTO'
        #     'xlim': [],
        #     'xlabel': '',
        #     'ylabel': '',
        #     'linewidth': 1.5,
        #     'fontweight': 'normal',
        #     'fontweight_ax': 'bold',
        #     'fontsize': 14,
        #     'fontsize_ax': 14,
        #     'fontname': 'DejaVu Sans Mono',
        #     'legend': 'off',
        #     'top_ticks': 'on',
        #     'spine_right': True,
        #     'spine_top': True,
        #     'spine_left': True,
        #     'spine_bottom': True,
        #     'size': (6, 6),
        #     'figsize':(6, 6),
        #     'dpi': 600,
        #     'grid': False,
        #     'tick_length': 6,
        #     'xticks':[],
        #     'yticks':[],
        # }
        default_props= {'linewidth':2.5,
                        'fontweight':'normal',
                        'fontweight_ax':'bold',
                        'fontsize':22,
                        'fontsize_ax':24,
                        'fontname':'Abyssinica SIL',#'DejaVu Sans Mono',
                        'legend':'off',
                        'top_ticks':'on',
                        'spine_right':True,
                        'spine_top':True,
                        'spine_left':True,
                        'spine_bottom':True,
                        'size':(6,6),
                        'dpi':600,
                        'grid':False,
                        'tick_length':6,
                        'xtick_format':[0,22,'center','top'],
                        'ytick_format':[0,22,'right','center'],
                        'xlim':'AUTO',
                        'ylim':'AUTO',
                        'xlabel':'',
                        'ylabel':''}
    
        if plot_props is None:
            plot_props = default_props.copy()
        else:
            for key, value in default_props.items():
                if key not in plot_props:
                    plot_props[key] = value
    
        return plot_props

def plot_grouped_values(data, groups,colors = None,spread=0.5,
                            showMedian=True,showMean=True,
                            plot_props=None,figsize=None,bins=10,
                            scatter_type = 'ordered',marker_size=50,
                            logY=False,logX=False):

        if plot_props is None:
            plot_props = instantiate_plot_props(plot_props)
        
        num_groups = len(groups)
        if isinstance(figsize, type(None)):
            figsize = (10,6)
            print('No size defined')
        
        # Create figure
        #print('figsize',figsize)
        fig,ax = plt.subplots(dpi=600,figsize=figsize)
        
        # Set some visual parameters
          # controls how much we jitter the points
        if isinstance(colors, type(None)):
            
            colors = np.random.random((num_groups,3))# Get distinct colors for each group
        
        if isinstance(colors,list):
            colors = np.asarray(colors)
        # Loop over each group
        for i, group in enumerate(groups):
            # Extract values for this group
            group_vals = data[:, i]
            
            group_vals = group_vals[np.invert(np.isnan(group_vals))]
             
            if scatter_type == 'ordered':
                counts, bin_edges = np.histogram(group_vals, bins=bins)
                bin_indices = np.digitize(group_vals, bin_edges) - 1
                x_vals = []
                y_vals = []
                bin_lengths = (counts-np.min(counts))/(np.max(counts)-np.min(counts))*spread
                
                unique_indices = np.unique(bin_indices)
                for b in unique_indices:
                     idxs_in_bin = np.where(bin_indices == b)[0]
                     n = len(idxs_in_bin)
                     if n > 0:
                         # Centered offsets around 0 → scaled to (-spread/2, +spread/2)
                         if n == 1:
                             x_offsets = np.asarray([0])
                         else:
                             x_offsets = np.linspace(-bin_lengths[b]/2, bin_lengths[b]/2, n)
            
                         
                         x_vals.extend(i + x_offsets+1)
                         y_vals.extend(group_vals[idxs_in_bin]) 

            
            elif scatter_type == 'random':

                kde = scipy.stats.gaussian_kde(group_vals)
                density_values = kde(group_vals)  # Get the density estimate for each value
                #print('density is',density_values)
                density_values = (density_values-np.min(density_values))/(
                                    np.max(density_values)-np.min(density_values))
                                    
                
                # Apply jitter proportional to the inverse of the density
                x_vals = (np.random.rand(len(group_vals)) - 0.5) * spread * density_values
                y_vals = group_vals
                x_vals = np.full_like(y_vals, i + 1) + x_vals
                
            # Plot mean as a horizontal line
          
            mean = np.mean(y_vals)
            med = np.median(y_vals)
                
            if showMean:
                
                
                ax.plot([i + 1 - 0.4, i + 1 + 0.4], 
                         [mean,mean], 
                         linewidth=4,zorder=1,color=[0, 0, 0, 0.6],
                         solid_capstyle='round')
            if showMedian:
                ax.plot([i + 1 - 0.3, i + 1 + 0.3], 
                         [med, med], 
                         linewidth=3,zorder=1,color=[0, 0, 0, 0.3],
                         solid_capstyle='round')
   
            # Scatter plot with jitter on x-axis
            x_vals = np.array(x_vals)
                      
            ax.scatter(x_vals, y_vals, 
                        color=colors[i,:], s=marker_size, alpha=0.6,zorder=3)
            
            
        
        # Customize the plot
        if not plot_props['ylim'] == 'AUTO' and len(plot_props['ylim']) > 0:

            ax.set_ylim(plot_props['ylim'])
            #print('set ylim',plot_props['ylim'])
        if not plot_props['xlim'] == 'AUTO' and len(plot_props['xlim']) > 0:
            ax.set_xlim(plot_props['xlim'])
            #print('set xlim',plot_props['xlim'])
        else:
            ax.set_xlim(0.5, num_groups + 0.5)
            
  
            
        ax.set_xlabel(plot_props['xlabel'])
        ax.set_ylabel(plot_props['ylabel'])
        if logY:
            ax.set_yscale("log", nonpositive='mask')
        if logX:
            ax.set_xscale("log", nonpositive='mask')
        ax.set_xticks(np.arange(1, num_groups + 1), groups, rotation=55)
        plt.gca().tick_params(width=2, labelsize=16, direction='out')

        style_plot(plt, ax,plot_props=plot_props)    
        plt.show()
        return (fig,ax)

def style_plot(plt, ax, plot_props=None,show=True):
        
        plt_prp = {'linewidth':2.5,
                        'fontweight':'normal',
                        'fontweight_ax':'bold',
                        'fontsize':22,
                        'fontsize_ax':24,
                        'fontname':'Abyssinica SIL',#'DejaVu Sans Mono',
                        'legend':'off',
                        'top_ticks':'on',
                        'spine_right':True,
                        'spine_top':True,
                        'spine_left':True,
                        'spine_bottom':True,
                        'size':(6,6),
                        'dpi':600,
                        'grid':False,
                        'tick_length':6,
                        'xtick_format':[0,22,'center','top'],
                        'ytick_format':[0,22,'right','center'],
                        'xlim':'AUTO',
                        'ylim':'AUTO',
                        'xlabel':'',
                        'ylabel':''}
    
    
        if plot_props is None:
            plot_props = {}
    
        # Helper function to get prop from plot_props or fallback to self.plt_prp
        def get_prop(key):
            return plot_props[key] if key in plot_props else plt_prp[key]
    
        # Line width for spines
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(get_prop('linewidth'))
    
        # Ticks
        if get_prop('top_ticks') == 'on':
            ax.tick_params(bottom=True, top=True)
            ax.tick_params(left=True, right=True)
    
        # Spines visibility
        ax.spines['right'].set_visible(get_prop('spine_right'))
        ax.spines['top'].set_visible(get_prop('spine_top'))
        ax.spines['left'].set_visible(get_prop('spine_left'))
        ax.spines['bottom'].set_visible(get_prop('spine_bottom'))
    
        # Tick width
        ax.tick_params(width=get_prop('linewidth'))
    
        # Legend
        if get_prop('legend') == 'on':
            plt.legend()
    
        # Axis font
        fontname = get_prop('fontname')
        ax.set_xlabel(ax.get_xlabel(), fontname=fontname)
        ax.set_ylabel(ax.get_ylabel(), fontname=fontname)
    
        # Tick label styling
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname(fontname)
            tick.set_fontsize(get_prop('fontsize'))
            tick.set_fontweight(get_prop('fontweight'))
    
        # Axis label size & weight
        params = {
            'axes.labelsize': get_prop('fontsize_ax'),
            'axes.labelweight': get_prop('fontweight_ax')
        }
        
        plt.rcParams.update(params)
    
        # Grid
        plt.grid(get_prop('grid'))
    
        # Tick length
        tick_length = get_prop('tick_length')
        ax.tick_params(axis='both', which='major', length=tick_length)
        ax.tick_params(axis='both', which='minor', length=tick_length * 0.6)
    
        # Plot formatting from self.plot_format
        
        
        for label in ax.get_xticklabels():

            label.set_rotation(get_prop("xtick_format")[0])
            label.set_fontsize(get_prop("xtick_format")[1])
            label.set_ha(get_prop("xtick_format")[2])   # or 'center', 'left'
            label.set_va(get_prop("xtick_format")[3])     # or 'center', 'bottom'
                
        for label in ax.get_yticklabels():
            label.set_rotation(get_prop("ytick_format")[0])
            label.set_fontsize(get_prop("ytick_format")[1])
            label.set_ha(get_prop("ytick_format")[2])   # or 'center', 'left'
            label.set_va(get_prop("ytick_format")[3])     # or 'center', 'bottom'
       
        
      
        # Additional settings that are only triggered if keys are explicitly present
        if 'y_axis_visible' in plot_props:
            ax.yaxis.set_visible(plot_props['y_axis_visible'])
            ax.spines['left'].set_visible(plot_props['y_axis_visible'])
    
        if 'x_axis_visible' in plot_props:
            ax.xaxis.set_visible(plot_props['x_axis_visible'])
            ax.spines['bottom'].set_visible(plot_props['x_axis_visible'])
    
        if 'x_scale' in plot_props:
            ax.set_xscale(plot_props['x_scale'])
            print('ADSfalfjlasdfj##############################################')
    
        if 'y_scale' in plot_props:
            ax.set_yscale(plot_props['y_scale'])
        if show:
            plt.show()



def class_histogram(
    data,
    label_classes,
    color=(0.2, 0.4, 0.8),   # RGB in [0,1]
    alpha=0.4,
    figsize=(6, 6),
    show_counts=False, 
    normalize = True,
    ax_labels = None,
    ):


    # Unique classes and counts
    if ax_labels is not None:
        u_class, cts = np.unique(data, return_counts=True)
        #all classes
        x = np.arange(len(ax_labels))
        # mapping found classes onto the vector
        cts2 = np.zeros_like(x)
        
        col = ax_labels.iloc[:,4].to_numpy()
        lookup = dict(zip(col, range(len(col))))
        mask = np.isin(u_class, col)
        loc = np.vectorize(lookup.get)(u_class, -1)
        cts2[loc] = cts
        cts = cts2
        u_class = col
                             
    else:
        u_class, cts = np.unique(data, return_counts=True)
        x = np.arange(len(u_class))

    if normalize:
        cts = cts/np.sum(cts)*100
    # Plot
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        x,
        cts,
        width=1.0,          # <-- touching bars
        color=color,
        alpha=alpha,
        edgecolor="black",
        linewidth=0
    )
 
    
    tf = np.isin(label_classes.iloc[:,4].to_numpy(),u_class)
    labels = label_classes.iloc[tf,0]
    # X-axis labels
    ax.set_xticks(x)
    print('b',labels)
    if labels is not None:
        if len(labels) != len(u_class):
            raise ValueError(
                f"labels length ({len(labels)}) must match number of classes ({len(u_class)})"
            )
        ax.set_xticklabels(labels)
    else:
        print('a')
        #ax.set_xticklabels(u_class)


    # Axis labels
    if normalize:
        ax.set_ylabel("Normalized count", fontsize=14)
    else: 
        ax.set_ylabel("Count", fontsize=14)
    
    ax.set_xlabel("Class", fontsize=14)
    
    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(2.0)
    
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=14,
        width=2,
        length=6
    )

    ax.margins(x=0)
    ax.tick_params(axis="x", labelrotation=90)


    # Annotate counts
    if show_counts:
        for rect, count in zip(bars, cts):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=10
            )

    plt.tight_layout()
    plt.show()
    
    return (fig,ax) 

###############################################################################
#    
# plot handling functions     
#
###############################################################################    
def add_mean_std_lines(fig, axes, mean, std,save=None):
    """
    Adds three horizontal lines to an existing matplotlib figure
    and shows the figure.
    """

    # Make this figure the active one
    plt.figure(fig.number)

    # Ensure axes is iterable
    if not isinstance(axes, (list, tuple)):
        axes = [axes]

    for ax in axes:
        # Preserve limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Add lines
        ax.axhline(mean, color='black', linestyle='-', linewidth=2, zorder=10)
        ax.axhline(mean + std, color='black', linestyle='--', linewidth=1, zorder=10)
        ax.axhline(mean - std, color='black', linestyle='--', linewidth=1, zorder=10)

        # Restore limits (avoid autoscale changes)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Redraw and show
    fig.canvas.draw()
    plt.show()
    
    if save is None:
        display(fig)
    else: 
        save_plot(fig,save[0],save[1])
        
    

def save_plot(axObj,name,s_path):
    if name is None or name  == '':
        return 
    if s_path is None or s_path == '':
        return 
    
    if not os.path.exists(s_path):
        os.mkdir(s_path)
        
    fname = s_path +'/' + name + '.png'
    pickle_name = s_path  +'/' + name + '.pickle'
    #print(fname)
    #print(pickle_name)
        
    if isinstance(axObj,tuple):
        axObj[0].savefig(fname, bbox_inches='tight')
        fig = axObj[0]
    else:
        axObj.savefig(fname, bbox_inches='tight')
        fig = axObj
    
    with open(pickle_name, 'wb') as f:
        pickle.dump(fig, f)
    
def load_plot(name,s_path):
    if not os.path.exists(s_path):
        os.mkdir(s_path)
    pickle_name = s_path + name + '.pickle'
    with open(pickle_name, 'rb') as f:
        fig = pickle.load(f)
    plt.figure(fig.number)
    plt.show()
    
    return fig,fig.axes      


def merge_pickled_barplots_over_ax1(name1, name2, s_path,save=None):
    """
    Load two pickled bar plots and draw all bars from plot2 onto plot1's axes.
    Returns fig1 and its axes.
    """
    # ---------- load figures ----------
    with open(os.path.join(s_path, name1 + '.pickle'), 'rb') as f:
        fig1 = pickle.load(f)
    with open(os.path.join(s_path, name2 + '.pickle'), 'rb') as f:
        fig2 = pickle.load(f)

    ax1 = fig1.axes[0]
    ax2 = fig2.axes[0]

    # ---------- collect bars ----------
    def get_bars(ax):
        return [p for p in ax.patches
                if isinstance(p, Rectangle) and p.get_width() > 0 and p.get_height() != 0]

    bars2 = get_bars(ax2)

    # ---------- draw bars from fig2 onto fig1 axes ----------
    for b in bars2:
        rect = Rectangle(
            b.get_xy(),
            b.get_width(),
            b.get_height(),
            facecolor=b.get_facecolor(),
            edgecolor=b.get_edgecolor(),
            linewidth=b.get_linewidth(),
            alpha=b.get_alpha(),
            transform=ax1.transData  # align with ax1 coordinates
        )
        ax1.add_patch(rect)

    # ---------- redraw figure ----------
    fig1.canvas.draw_idle()
    fig1.canvas.draw()
    plt.show()
    
    if save is None:
        display(fig1)
    else: 
        save_plot(fig1,save[0],save[1])
    
    return fig1, ax1