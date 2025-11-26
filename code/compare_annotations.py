#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 18:15:26 2025

@author: christian
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from IPython import embed
from measurements import find_head_tail,get_worm_centerline,measure_tube_thickness



def load_coco_annotations(json_file):
    """Load COCO-style annotations and return polygons and centers."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    polygons = []
    centers = []
    
    for anno in data['annotations']:
        # COCO 'segmentation' can be list of polygons
        
        
        if isinstance(anno['segmentation'], list):
            for seg in anno['segmentation']:
                poly = np.array(seg).reshape(-1, 2)
                polygons.append(poly)
                center = np.mean(poly, axis=0)
                centers.append(center)
        else:
            # could handle RLE if needed
            pass
    
    return polygons, np.array(centers)

def match_annotations(json_a, json_b, cutoff=20.0, debug=True):
    """Match annotations between two COCO JSON files based on center distances."""
    print("DRECKSHUR")
    polys_a, centers_a = load_coco_annotations(json_a)
    polys_b, centers_b = load_coco_annotations(json_b)
   
    matches = []  # list of tuples (idx_a, idx_b)
    
    # Compute distances
    over_polies = []
    for i, ca in enumerate(centers_a):
        for j, cb in enumerate(centers_b):
            dist = np.linalg.norm(ca - cb)
            if dist <= cutoff:
                # Check if polygons overlap
                poly_a = Polygon(polys_a[i])
                poly_b = Polygon(polys_b[j])
                if poly_a.intersects(poly_b):
                    matches.append((i, j))
                    over_polies.append([poly_a,poly_b])
    
    if debug:
        print(f"Found {len(matches)} matching annotations")
    
    # ---------------------------
    # Plot overlay
    # ---------------------------

    residuals = []
    # plot all A annotations
    for polies in over_polies:
 
        fig, ax = plt.subplots(figsize=(8,8),dpi=600)
        poly_a = polies[0]
        poly_b = polies[1]
        #ax.fill(poly[:,0], poly[:,1], facecolor='blue', alpha=0.3, edgecolor='blue')
    
    # plot all B annotations
  
        #ax.fill(poly[:,0], poly[:,1], facecolor='red', alpha=0.3, edgecolor='red')
    

        
        xa, ya = poly_a.exterior.coords.xy
        xb, yb = poly_b.exterior.coords.xy
        ax.plot(xa, ya, color='blue', lw=2,alpha=0.3)
        ax.plot(xb, yb, color='red', lw=2,alpha=0.3)
    
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()

        coords1 = np.zeros((len(xa),2))
        coords1[:,0] = xa
        coords1[:,1] = ya
        
        coords2 = np.zeros((len(xb),2))
        coords2[:,0] = xb
        coords2[:,1] = yb
        
        m1 = do_measurements(coords1)
        m2 = do_measurements(coords2)
        del m1['area']
        del m2['area']
        
        dicts = [m1,m2]
        keys = m1.keys()
        deltas = [m1[k] - m2[k] for k in m1.keys()]
        
        # plot
        fig, ax = plt.subplots()
        ax.bar(keys, deltas)
        ax.set_ylabel('Δ (max - min)')
        ax.set_title('Value Differences per Key')
        
        # reliable: set rotation via tick_params
        ax.tick_params(axis='x', labelrotation=90)
        
        plt.tight_layout()
        plt.show()
          
    return matches

def do_measurements(coords):

    head_coords, tail_coords = find_head_tail(coords, 1, int(coords.shape[0]/10), debug=True)
    
    centerline,length,area = get_worm_centerline(coords, plot=True, padding=10)
    percents=np.arange(0.05,1,0.05)
    
    
   
    
    thicks,length = measure_tube_thickness(coords,centerline,head_coords,tail_coords,percents = percents, debug=True)
    output = {}

    for i in percents:
        
        output['percent_'+str(i)] = thicks[i]
        output["length"] = length
        output["area"] = area
    return output


# ---------------------------
# Example usage:
# ---------------------------
# matches = match_annotations('fileA.json', 'fileB.json', cutoff=25.0)
