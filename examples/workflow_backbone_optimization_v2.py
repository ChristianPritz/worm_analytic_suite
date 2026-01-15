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
from matplotlib.path import Path

def centerline_inside_polygon(centerline, polygon):
    """
    Check if centerline points lie inside polygon.

    Returns
    -------
    inside_all : int (1 if all inside, else 0)
    inside_fraction : float in [0,1]
    """
    poly_path = Path(polygon)
    inside = poly_path.contains_points(centerline)

    inside_fraction = np.mean(inside)
    inside_all = int(inside_fraction == 1.0)

    return inside_all, inside_fraction

def circular_interpolate(scores, valid_idx, N):
    """
    Interpolate sparse circular scores onto full polygon.
    """
    full = np.zeros(N)

    # duplicate for circularity
    x = np.concatenate([valid_idx, valid_idx + N])
    y = np.concatenate([scores, scores])

    xi = np.arange(2*N)
    yi = np.interp(xi, x, y)

    return yi[:N]
def pick_from_score_map(score_map, num=5, min_dist=10):
    """
    Pick top-N peaks from a circular score map.
    """
    idx = np.argsort(score_map)[::-1]
    selected = []

    for i in idx:
        if all(min((i - j) % len(score_map),
                   (j - i) % len(score_map)) > min_dist
               for j in selected):
            selected.append(i)
        if len(selected) >= num:
            break

    return np.array(selected)

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

        inside_all, inside_fraction = centerline_inside_polygon(
            centerline, polygon
        )

        return {
            "centerline": centerline,
            "length": length,
            "area": area,
            "curvature": curvature,
            "mean_smoothness": np.mean(curvature),
            "inside_all": inside_all,               # 0 or 1
            "inside_fraction": inside_fraction,     # optional
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
    w_inside=5.0,      # STRONG penalty weight
    peak_q=1
):
    lengths = np.array([c["length"] for c in cands], dtype=float)

    mean_bends = np.array([
        np.mean(c["curvature"]) for c in cands
    ])

    peak_bends = np.array([
        np.quantile(c["curvature"], peak_q) for c in cands
    ])

    inside_all = np.array([
        c["inside_all"] for c in cands
    ], dtype=float)

    inside_frac = np.array([
        c["inside_fraction"] for c in cands
    ], dtype=float)

    # Normalize geometric terms
    L = (lengths - lengths.min()) / (np.ptp(lengths) + 1e-8)
    M = (mean_bends - mean_bends.min()) / (np.ptp(mean_bends) + 1e-8)
    P = (peak_bends - peak_bends.min()) / (np.ptp(peak_bends) + 1e-8)

    # Base geometric score
    base_score = w_length * L - w_mean * M - w_peak * P

    # Containment penalty (VERY STRONG)
    containment_penalty = w_inside * (1.0 - inside_frac)

    scores = base_score - containment_penalty

    for c, s in zip(cands, scores):
        c["score"] = s
        c["mean_bend"] = np.mean(c["curvature"])
        c["peak_bend"] = np.quantile(c["curvature"], peak_q)

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
    weights
):
    poly = np.asarray(polygon)
    N = len(poly)

    candidate_db = []

    # ============================================================
    # PHASE 1 — COARSE GLOBAL MAPPING
    # ============================================================
    print("Phase 1: global score map")

    sampled_heads = np.linspace(0, N-1, coarse_params[0]["num_start_points"], dtype=int)
    scores = []
    valid_idx = []

    for h in sampled_heads:
        t = (h + N//2) % N
        res = evaluate_candidate(poly, h, t, backbone_func, weights)
        if res is None:
            continue

        scored = score_candidates(
            [res],
            w_length=weights["length"],
            w_mean=weights["mean_smooth"],
            w_peak=weights["max_bend"]
        )

        candidate_db.append(scored)
        scores.append(scored["score"])
        valid_idx.append(h)

    scores = np.array(scores)
    valid_idx = np.array(valid_idx)

    head_score_map = circular_interpolate(scores, valid_idx, N)

    # ============================================================
    # PHASE 2 — GRADIENT-BASED HEAD REFINEMENT
    # ============================================================
    print("Phase 2: head refinement")

    head_candidates = pick_from_score_map(
        head_score_map,
        num=branch_params[0]["num_start_points"]
    )

    for h in head_candidates:
        for dt in np.linspace(-branch_params[0]["search_space"],
                              branch_params[0]["search_space"],
                              branch_params[0]["end_points_each"]).astype(int):

            t = (h + N//2 + dt) % N
            res = evaluate_candidate(poly, h, t, backbone_func, weights)
            if res is None:
                continue

            scored = score_candidates(
                [res],
                w_length=weights["length"],
                w_mean=weights["mean_smooth"],
                w_peak=weights["max_bend"]
            )

            candidate_db.append(scored)

    # ============================================================
    # PHASE 3 — FINAL REFINEMENT USING UPDATED MAPS
    # ============================================================
    print("Phase 3: final refinement")

    # rebuild head score map from database
    head_idx = np.array([c["head_idx"] for c in candidate_db])
    head_scores = np.array([c["score"] for c in candidate_db])

    head_score_map = circular_interpolate(head_scores, head_idx, N)

    fine_heads = pick_from_score_map(
        head_score_map,
        num=branch_params[1]["num_start_points"],
        min_dist=3
    )

    for h in fine_heads:
        tail_scores = []
        tail_idx = []

        for dt in np.linspace(-branch_params[1]["search_space"],
                              branch_params[1]["search_space"],
                              branch_params[1]["end_points_each"]).astype(int):

            t = (h + N//2 + dt) % N
            res = evaluate_candidate(poly, h, t, backbone_func, weights)
            if res is None:
                continue

            scored = score_candidates(
                [res],
                w_length=weights["length"],
                w_mean=weights["mean_smooth"],
                w_peak=weights["max_bend"]
            )

            candidate_db.append(scored)
            tail_scores.append(scored["score"])
            tail_idx.append(t)

    # ============================================================
    # FINAL WINNER FROM DATABASE
    # ============================================================
    best = max(candidate_db, key=lambda x: x["score"])
    return best["centerline"], best, candidate_db

def select_candidate(
    candidate_db,
    key,
    mode="max",
    value=None,
    q=None
):
    """
    Select a candidate from candidate_db based on a metric.

    Parameters
    ----------
    candidate_db : list of dict
    key : str
        Metric name (e.g. "score", "length", "peak_bend", "inside_fraction")
    mode : {"max", "min", "quantile", "closest"}
    value : float, optional
        Used for mode="closest"
    q : float in [0,1], optional
        Used for mode="quantile"

    Returns
    -------
    candidate : dict
    idx : int
    """

    values = np.array([c[key] for c in candidate_db], dtype=float)

    if mode == "max":
        idx = np.argmax(values)

    elif mode == "min":
        idx = np.argmin(values)

    elif mode == "quantile":
        if q is None:
            raise ValueError("q must be provided for quantile mode")
        target = np.quantile(values, q)
        idx = np.argmin(np.abs(values - target))

    elif mode == "closest":
        if value is None:
            raise ValueError("value must be provided for closest mode")
        idx = np.argmin(np.abs(values - value))

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    return candidate_db[idx], idx

def plot_candidate(
    polygon,
    candidate,
    ax=None,
    title=None,
    show_head_tail=True,
    lw_poly=1.5,
    lw_center=2.5
):
    """
    Plot polygon and candidate centerline.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    poly = np.asarray(polygon)
    cl = np.asarray(candidate["centerline"])

    ax.plot(poly[:, 0], poly[:, 1], "-k", lw=lw_poly, label="polygon")
    ax.plot(cl[:, 0], cl[:, 1], "-r", lw=lw_center, label="centerline")

    if show_head_tail:
        ax.scatter(*poly[candidate["head_idx"]], c="g", s=60, label="head")
        ax.scatter(*poly[candidate["tail_idx"]], c="b", s=60, label="tail")

    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title)

    ax.legend()
    return ax

def candidate_db_to_array(
    candidate_db,
    include_keys=None,
    exclude_keys=None,
    allow_bool=True
):
    """
    Convert candidate_db to a numeric NumPy array using scalar-valued keys only.

    Parameters
    ----------
    candidate_db : list of dict
    include_keys : list of str, optional
        Explicit keys to include (others ignored)
    exclude_keys : list of str, optional
        Keys to exclude
    allow_bool : bool
        If True, booleans are converted to 0/1

    Returns
    -------
    X : ndarray (n_candidates, n_metrics)
    keys : list of str
        Column names corresponding to X
    """

    if not candidate_db:
        raise ValueError("candidate_db is empty")

    # Determine usable keys from first candidate
    first = candidate_db[0]

    scalar_keys = []
    for k, v in first.items():

        if include_keys is not None and k not in include_keys:
            continue
        if exclude_keys is not None and k in exclude_keys:
            continue

        # Accept scalar numeric values
        if np.isscalar(v):
            if isinstance(v, bool) and not allow_bool:
                continue
            scalar_keys.append(k)

    if not scalar_keys:
        raise ValueError("No scalar-valued keys found")

    # Build array
    X = np.zeros((len(candidate_db), len(scalar_keys)), dtype=float)

    for i, c in enumerate(candidate_db):
        for j, k in enumerate(scalar_keys):
            val = c.get(k, np.nan)
            X[i, j] = float(val)

    return X, scalar_keys

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
    {"num_start_points": 4},
    {"num_start_points": 6, "search_space": int(400/4)}
]

branch_params = [
    {"num_start_points": 10, "end_points_each": 3, "search_space": 20},
    {"num_start_points": 10, "end_points_each": 5, "search_space": 5}
]
weights = {"length": 2.0, "mean_smooth": .5,"max_bend":1.0}

 
   
output = optimize_backbone(
    area_coords2,
    get_worm_centerline_sliding,
    coarse_params,
    branch_params,
    weights=weights
)

fig, ax = plt.subplots()
ax.plot(area_coords2[:,0],area_coords2[:,1])
ax.plot(output[1]["centerline"][:,0],output[1]["centerline"][:,1])
plt.show()

cands,keys = candidate_db_to_array(output[2])
length = 0 
area = 1
mean_smoothness = 2
inside_all=3
score = 7

idx = np.where(cands[:,length] == np.max(cands[:,length]))

plot_candidate(
    area_coords2,
    output[2][int(idx[0])],
    ax=None,
    title=None,
    show_head_tail=True,
    lw_poly=1.5,
    lw_center=2.5
)



idx = np.where(cands[:,score] == np.max(cands[:,score]))

plot_candidate(
    area_coords2,
    output[2][int(idx[0])],
    ax=None,
    title=None,
    show_head_tail=True,
    lw_poly=1.5,
    lw_center=2.5
)


