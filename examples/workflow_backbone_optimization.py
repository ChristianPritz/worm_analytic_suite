#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:52:37 2026

@author: wormulon
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
import pandas as pd
import cv2
from measurements import *
from worm_plotter import worm_width_plot, df_to_grouped_array,plot_grouped_values,class_histogram
from classifiers import  classify,run_kmeans_and_show,label_coco_areas,coco_areas_calculation,train_classifier
from annotation_tool_v8 import AnnotationTool
from sklearn.decomposition import PCA
    
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from shapely.geometry import Polygon, LineString
from shapely.ops import nearest_points
from scipy.spatial import ConvexHull
import numpy as np

def backbone_smoothness(centerline):
    """
    Compute smoothness via mean squared curvature.
    Lower = smoother.
    """
    cl = np.asarray(centerline)
    if len(cl) < 5:
        return np.inf

    d1 = np.gradient(cl, axis=0)
    d2 = np.gradient(d1, axis=0)

    curvature = np.linalg.norm(
        d1[:, 0]*d2[:, 1] - d1[:, 1]*d2[:, 0]
    ) / (np.linalg.norm(d1, axis=1)**3 + 1e-8)

    return np.nanmean(curvature**2)

def evaluate_candidate(polygon, h_idx, t_idx, backbone_func, weights):
    try:
        head = polygon[h_idx]
        tail = polygon[t_idx]

        centerline, length, area = backbone_func(
            polygon, head, tail,
            window_size=5,
            multiplier=1,
            smooth_win=15,
            smooth_poly=3,
            debug=True
        )

        if centerline is None or len(centerline) < 5:
            return None

        curvature = compute_curvature(centerline)

        return {
            "centerline": centerline,
            "length": length,
            "area": area,
            "curvature": curvature,
            "mean_smoothness": np.mean(curvature),
            "head_idx": h_idx,
            "tail_idx": t_idx,
        }

    except Exception:
        return None

# def score_candidates(cands, w_length=1.0, w_smooth=1.0):
#     lengths = np.array([c["length"] for c in cands], dtype=float)
#     smooths = np.array([c["smoothness"] for c in cands], dtype=float)

#     # Normalize safely (NumPy 2.0 compatible)
#     L_range = np.ptp(lengths)
#     S_range = np.ptp(smooths)

#     L = (lengths - lengths.min()) / (L_range + 1e-8)
#     S = (smooths - smooths.min()) / (S_range + 1e-8)

#     scores = w_length * L - w_smooth * S

#     for c, s in zip(cands, scores):
#         c["score"] = s

#     return max(cands, key=lambda x: x["score"])


def score_candidates(
    cands,
    w_length=1.0,
    w_mean=0.3,
    w_peak=1.0,
    peak_q=0.99
):
    lengths = np.array([c["length"] for c in cands], dtype=float)

    mean_bends = np.array([
        np.mean(c["curvature"])
        for c in cands
    ])

    peak_bends = np.array([
        np.quantile(c["curvature"], peak_q)
        for c in cands
    ])

    # Normalize
    L = (lengths - lengths.min()) / (np.ptp(lengths) + 1e-8)
    M = (mean_bends - mean_bends.min()) / (np.ptp(mean_bends) + 1e-8)
    P = (peak_bends - peak_bends.min()) / (np.ptp(peak_bends) + 1e-8)

    scores = w_length * L - w_mean * M - w_peak * P

    for c, s in zip(cands, scores):
        c["score"] = s
        c["mean_bend"] = np.mean(c["curvature"])
        c["peak_bend"] = np.quantile(c["curvature"], peak_q)

    print(scores,L,M,P)
    return max(cands, key=lambda x: x["score"])

def compute_curvature(centerline, eps=1e-8):
    """
    Discrete curvature for a 2D polyline.
    Returns per-point curvature magnitude.
    """
    cl = np.asarray(centerline)

    # First derivatives
    dx = np.gradient(cl[:, 0])
    dy = np.gradient(cl[:, 1])

    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula
    kappa = np.abs(dx * ddy - dy * ddx) / (dx*dx + dy*dy + eps)**1.5
    return np.nan_to_num(kappa)


def optimize_backbone(
    polygon,
    backbone_func,
    coarse_params,
    branch_params,
    weights={"length": 1.0, "smooth": 1.0}
):
    """
    Hierarchical optimization of worm backbone endpoints.

    Parameters
    ----------
    polygon : (N,2) ndarray
    backbone_func : callable
    coarse_params : list of dicts
    branch_params : list of dicts
    weights : dict
        {"length": w1, "smooth": w2}

    Returns
    -------
    best_centerline : (L,2) ndarray
    best_meta : dict
    """

    poly = np.asarray(polygon)
    N = len(poly)

    # ---------- COARSE + FINE LOOP ----------
    current_best = None
    candidate_heads = np.arange(N)
    print("loop1")
    for stage in coarse_params:
        num = stage["num_start_points"]
        search = stage.get("search_space", N)

        center = current_best["head_idx"] if current_best else 0
        half = search // 2

        idxs = (center + np.linspace(-half, half, num).astype(int)) % N

        candidates = []
        for h in idxs:
            t = (h + N//2) % N
            res = evaluate_candidate(poly, h, t, backbone_func, weights)
            if res:
                candidates.append(res)

        if not candidates:
            continue

        current_best = score_candidates(
            candidates,
            w_length=weights["length"],
            w_mean=weights["mean_smooth"],
            w_peak=weights["max_bend"]
        )

    # ---------- BRANCHING LOOP ----------
    print("loop2")
    for stage in branch_params:
        num_h = stage["num_start_points"]
        num_t = stage["end_points_each"]
        search = stage["search_space"]

        h_center = current_best["head_idx"]
        t_center = current_best["tail_idx"]

        h_idxs = (h_center + np.linspace(-search, search, num_h).astype(int)) % N
        t_offsets = np.linspace(-search, search, num_t).astype(int)

        candidates = []
        for h in h_idxs:
            for dt in t_offsets:
                t = (h + N//2 + dt) % N
                res = evaluate_candidate(poly, h, t, backbone_func, weights)
                if res:
                    candidates.append(res)

        if not candidates:
            continue

        current_best = score_candidates(
            candidates,
            w_length=weights["length"],
            w_mean=weights["mean_smooth"],
            w_peak=weights["max_bend"]
        )

    return current_best["centerline"], current_best

## werk: 
image_dir = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images' # this is where the images are sitting
point_csv =  '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_points.csv'
area_json =  '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_areas.json'
color_csv =  '/home/wormulon/models/worm_analytic_suite/class_colors.csv'






ids, bbs,segs,ls,ars = load_metrics_from_json(area_json)
circs = np.array(compute_circumferences(segs))
kappas = backbone_curvature_stats(bbs)
kappas2 = backbone_curvature_stats(segs)
a,loop_count,c = detect_loops_batch(bbs)
compute_polygon_bulbiness_batch(segs,bbs)
bulbiness = compute_polygon_bulbiness_batch(segs,bbs)
conc,conv = batch_concavity_convexity(segs)

area_coords = np.array(segs[0][0])
area_coords = area_coords.reshape(-1,2)
area_coords1 = upsample_closed_contour(area_coords, N_min=400)
area_coords2 = smooth_polygon_gaussian(area_coords1, sigma=4.0, closed=True)

coarse_params = [
    {"num_start_points": 6},
    {"num_start_points": 6, "search_space": int(400/6)}
]

branch_params = [
    {"num_start_points": 10, "end_points_each": 3, "search_space": 20},
    {"num_start_points": 10, "end_points_each": 5, "search_space": 5}
]
weights = {"length": 1.0, "mean_smooth": .2,"max_bend":1.0}

 
   
output = optimize_backbone(
    area_coords2,
    get_worm_centerline_sliding,
    coarse_params,
    branch_params,
    weights=weights
)

fig, ax = plt.subplots()
ax.plot(area_coords2[:,0],area_coords2[:,1])
ax.plot(output["centerline"][:,0],output["centerline"][:,1])
plt.show()



