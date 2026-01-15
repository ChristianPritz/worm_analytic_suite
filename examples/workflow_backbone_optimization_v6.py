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
from shapely.geometry import Polygon, Point
from IPython import embed

def centerline_inside_polygon(centerline, polygon):
    """
    Check if centerline points lie inside polygon using Shapely.

    Parameters
    ----------
    centerline : (N,2) array
        Points along the centerline
    polygon : (M,2) array
        Polygon coordinates (can be concave, complex)

    Returns
    -------
    inside_all : int
        1 if all centerline points are inside polygon, else 0
    inside_fraction : float
        Fraction of centerline points inside polygon [0,1]
    """
    # Create Shapely polygon (automatically closed if needed)
    poly = Polygon(polygon)

    inside_flags = np.array([poly.contains(Point(pt)) for pt in centerline], dtype=bool)

    inside_fraction = inside_flags.mean()
    inside_all = int(inside_flags.all())

    return inside_all, inside_fraction


def end_curvature_ratio(centerline, frac=0.05, eps=1e-8):
    """
    Ratio of end curvature to global curvature.
    Lower = smoother ends.
    """
    curvature = compute_curvature(centerline)

    n = len(curvature)
    k = max(1, int(np.ceil(frac * n)))

    end_curv = np.concatenate([curvature[:k], curvature[-k:]])
    mean_end = np.mean(end_curv)
    mean_all = np.mean(curvature)

    return mean_end / (mean_all + eps)


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
        centerline, length, area = backbone_func(
            polygon, h_idx, t_idx,
            window_size=5,
            multiplier=1,
            smooth_win=15,
            smooth_poly=3,
            debug=True
        )

        if centerline is None or len(centerline) < 5:
            return None

        curvature = compute_curvature(centerline)
        mean_bend = np.mean(curvature)
        peak_bend = np.quantile(curvature, 1.0)

        # NEW metric
        end_curv_ratio = end_curvature_ratio(centerline, frac=0.05)

        inside_all, inside_fraction = centerline_inside_polygon(
            centerline, polygon
        )

        return {
            "centerline": centerline,
            "length": length,
            "area": area,
            "curvature": curvature,
            "mean_smoothness": mean_bend,
            "peak_bend": peak_bend,
            "end_curvature_ratio": end_curv_ratio,  # ← NEW
            "inside_all": inside_all,
            "inside_fraction": inside_fraction,
            "head_idx": h_idx,
            "tail_idx": t_idx,
        }

    except Exception:
        return None


def score_candidates_scaled(cands, metric_keys, weights, min_max=None):
    """
    Score candidates using min-max scaling (0–1) instead of z-score.

    Parameters
    ----------
    cands : list of dict
        Each dict has scalar metrics.
    metric_keys : list of str
        Keys to consider for scoring
    weights : dict
        Weight per metric (must match metric_keys)
    min_max : dict, optional
        Precomputed min and max per metric (from Phase 1)
        If None, compute from this candidate list.

    Returns
    -------
    best_candidate : dict
        Candidate with highest weighted score
    X_scaled : ndarray
        Min-max scaled metric matrix (n_candidates x n_metrics)
    min_max : dict
        {'metric1': (min, max), ...}
    """
    # Build metric matrix
    X = np.zeros((len(cands), len(metric_keys)), dtype=float)
    for i, c in enumerate(cands):
        for j, k in enumerate(metric_keys):
            X[i, j] = float(c[k])

    # Compute min-max scaling
    if min_max is None:
        min_max = {}
        X_scaled = np.zeros_like(X)
        for j, k in enumerate(metric_keys):
            print(j,k)
            print(X)
            mn = np.min(X[:, j])
            mx = np.max(X[:, j])
            X_scaled[:, j] = (X[:, j] - mn) / (mx - mn + 1e-8)
            min_max[k] = (mn, mx)
    else:
        X_scaled = np.zeros_like(X)
        for j, k in enumerate(metric_keys):
            mn, mx = min_max[k]
            X_scaled[:, j] = (X[:, j] - mn) / (mx - mn + 1e-8)

    # Weighted sum of scaled metrics
    w = np.array([weights.get(k, 0.0) for k in metric_keys])
    
    scores = X_scaled @ w

    # Assign scores to candidates
    for c, s, scaled in zip(cands, scores, X_scaled):
        c["score"] = s
        c["scaled_metrics"] = dict(zip(metric_keys, scaled))

    best_idx = np.argmax(scores)
    return cands[best_idx], X_scaled, min_max

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

def optimize_backbone_scaled(
    polygon,
    backbone_func,
    coarse_params,
    branch_params,
    metric_keys,
    weights,
    debug = True
):
    """
    Hierarchical backbone optimization with min-max scaled scoring.
    Phase 2 now varies both head and tail points around the current best candidate in two steps:
    1. Random coarse variations
    2. Finer-grained local variations
    """

    poly = np.asarray(polygon)
    N = len(poly)
    candidate_db = []
    # ===============================
    # PHASE 0 - head tail determination
    # ===============================
    print("Phase 0: head-tail initialization")
    
    h0, t0 = find_head_tail_robust(poly, win_frac=0.05, debug=debug)
    
    # Convert head/tail coordinates → nearest polygon indices
    def closest_idx(pt, poly):
        return np.argmin(np.linalg.norm(poly - pt[None, :], axis=1))
    
    h0_idx = closest_idx(h0, poly)
    t0_idx = closest_idx(t0, poly)
    
    # Evaluate Phase-0 candidate
    res0 = evaluate_candidate(poly, h0_idx, t0_idx, backbone_func, weights)
    
    if res0 is not None:
        res0["origin"] = "phase0_headtail"
        candidate_db.append(res0)

    # ===============================
    # PHASE 1 — COARSE GLOBAL MAPPING
    # ===============================
    print("Phase 1: global mapping")

    sampled_heads = np.linspace(0, N-1, coarse_params[0]["num_start_points"], dtype=int)
    for h in sampled_heads:
        t = (h + N//2) % N
        res = evaluate_candidate(poly, h, t, backbone_func, weights)
        if res is not None:
            candidate_db.append(res)

    # Phase 1 scoring using min-max scaling
    best, X_scaled, min_max = score_candidates_scaled(candidate_db, metric_keys, weights, min_max=None)

    # ===============================
    # PHASE 2 — FINE TUNING / BRANCHING
    # ===============================
    print("Phase 2: fine-tune")

    # Step 1: Coarse local variations around first-phase winner
    coarse_range = branch_params[0]["search_space"]
    coarse_steps = branch_params[0]["end_points_each"]

    best_head = best["head_idx"]
    best_tail = best["tail_idx"]

    for _ in range(coarse_steps):
        # Random variation within +/- coarse_range
        delta_h = np.random.randint(-coarse_range, coarse_range + 1)
        delta_t = np.random.randint(-coarse_range, coarse_range + 1)

        h_new = (best_head + delta_h) % N
        t_new = (best_tail + delta_t) % N

        res = evaluate_candidate(poly, h_new, t_new, backbone_func, weights)
        if res is None:
            continue

        # Re-score using Phase 1 min-max
        best_temp, X_scaled_temp, _ = score_candidates_scaled(
            [res], metric_keys, weights, min_max=min_max
        )

        candidate_db.append(best_temp)

    # Update best after coarse step
    best = max(candidate_db, key=lambda x: x["score"])
    best_head = best["head_idx"]
    best_tail = best["tail_idx"]

    # Step 2: Finer-grained variations around updated best
    fine_range = branch_params[1]["search_space"]
    fine_steps = branch_params[1]["end_points_each"]

    for dt_h in np.linspace(-fine_range, fine_range, fine_steps).astype(int):
        for dt_t in np.linspace(-fine_range, fine_range, fine_steps).astype(int):
            h_new = (best_head + dt_h) % N
            t_new = (best_tail + dt_t) % N

            res = evaluate_candidate(poly, h_new, t_new, backbone_func, weights)
            if res is None:
                continue

            # Re-score using Phase 1 min-max
            best_temp, X_scaled_temp, _ = score_candidates_scaled(
                [res], metric_keys, weights, min_max=min_max
            )

            candidate_db.append(best_temp)

    # ===============================
    # FINAL WINNER
    # ===============================
    best_final = max(candidate_db, key=lambda x: x["score"])
    return best_final["centerline"], best_final, candidate_db, min_max



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
    Robust to mixed output structures (arrays, strings, metadata).
    """

    if not candidate_db:
        raise ValueError("candidate_db is empty")

    # ------------------------------------
    # Determine candidate keys to consider
    # ------------------------------------
    all_keys = set().union(*(c.keys() for c in candidate_db))

    scalar_keys = []

    for k in all_keys:

        if include_keys is not None and k not in include_keys:
            continue
        if exclude_keys is not None and k in exclude_keys:
            continue

        is_scalar = True

        for c in candidate_db:
            v = c.get(k, None)

            # Missing → NaN is OK
            if v is None:
                continue

            # Reject non-scalars
            if not np.isscalar(v):
                is_scalar = False
                break

            # Reject bools if not allowed
            if isinstance(v, bool) and not allow_bool:
                is_scalar = False
                break

            # Reject strings explicitly
            if isinstance(v, str):
                is_scalar = False
                break

        if is_scalar:
            scalar_keys.append(k)

    if not scalar_keys:
        raise ValueError("No scalar-valued numeric keys found")

    # ------------------------------------
    # Build numeric array
    # ------------------------------------
    X = np.full((len(candidate_db), len(scalar_keys)), np.nan, dtype=float)

    for i, c in enumerate(candidate_db):
        for j, k in enumerate(scalar_keys):
            v = c.get(k, np.nan)
            if v is None:
                continue
            X[i, j] = float(v)

    return X, scalar_keys


def upsample_closed_contour(contour, N_min=100):
    """
    Upsample a closed polygonal contour to at least N_min points
    using arc-length interpolation.

    Parameters
    ----------
    contour : np.ndarray (N,2)
        Ordered polygon vertices (may or may not be explicitly closed)
    N_min : int
        Minimum number of points after upsampling

    Returns
    -------
    contour_up : np.ndarray (M+1,2)
        Upsampled closed contour (first point repeated at end)
    """

    c = np.asarray(contour, dtype=float)

    if c.shape[0] < 3:
        raise ValueError("Contour must have at least 3 points")

    # --- Ensure closed contour ---
    if not np.allclose(c[0], c[-1]):
        c = np.vstack([c, c[0]])

    # --- Arc-length parameterization ---
    diffs = np.diff(c, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0], np.cumsum(seglen)])
    total_len = s[-1]

    if total_len == 0:
        raise ValueError("Degenerate contour with zero length")

    # --- Decide target number of points ---
    N_orig = len(c) - 1  # exclude duplicate
    N_target = max(N_min, N_orig)

    s_new = np.linspace(0, total_len, N_target + 1)

    # --- Interpolation ---
    fx = interp1d(s, c[:, 0], kind="linear")
    fy = interp1d(s, c[:, 1], kind="linear")

    x_new = fx(s_new)
    y_new = fy(s_new)

    contour_up = np.column_stack((x_new, y_new))

    # --- Enforce exact closure ---
    contour_up[-1] = contour_up[0]

    return contour_up


def smooth_polygon_gaussian(contour, sigma=2.0, closed=True):
    """
    Smooth a polygon by Gaussian filtering along the contour.

    Parameters
    ----------
    contour : (N,2) ndarray
        Ordered polygon points
    sigma : float
        Smoothing strength (in points)
    closed : bool
        Whether contour is closed

    Returns
    -------
    smooth_contour : (N,2) ndarray
    """

    c = np.asarray(contour)

    mode = "wrap" if closed else "nearest"

    xs = scipy.ndimage.gaussian_filter1d(c[:, 0], sigma, mode=mode)
    ys = scipy.ndimage.gaussian_filter1d(c[:, 1], sigma, mode=mode)

    return np.column_stack((xs, ys))

def detect_loops_batch(backbones):
    """
    Detect loops for multiple 2D backbones.

    Parameters
    ----------
    backbones : np.ndarray, shape (n, l, 2)
        Array of n backbones, each with l points (x, y)

    Returns
    -------
    has_loop : np.ndarray, shape (n,)
        True if backbone has at least one loop
    loop_counts : np.ndarray, shape (n,)
        Number of intersecting segment pairs per backbone
    loop_indices_list : list of lists of tuples
        Indices of intersecting segments for each backbone
    """

    n = backbones.shape[0]
    has_loop = np.zeros(n, dtype=bool)
    loop_counts = np.zeros(n, dtype=int)
    loop_indices_list = []

    def segments_intersect(p1, p2, q1, q2):
        """Check if two line segments (p1,p2) and (q1,q2) intersect."""
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return (ccw(p1,q1,q2) != ccw(p2,q1,q2)) and (ccw(p1,p2,q1) != ccw(p1,p2,q2))

    for i in range(n):
        bb = backbones[i]
        L = bb.shape[0]
        loop_indices = []

        if np.isnan(bb).all() or L < 4:
            # not enough points to form a loop
            loop_indices_list.append(loop_indices)
            continue

        # check each segment against all non-adjacent segments
        for s1 in range(L-1):
            p1, p2 = bb[s1], bb[s1+1]
            for s2 in range(s1+2, L-1):
                q1, q2 = bb[s2], bb[s2+1]
                if segments_intersect(p1, p2, q1, q2):
                    loop_indices.append((s1, s2))

        has_loop[i] = len(loop_indices) > 0
        loop_counts[i] = len(loop_indices)
        loop_indices_list.append(loop_indices)

    return has_loop, loop_counts, loop_indices_list




#window_size  = 5
def trace_centerline(poly, idx_head, idx_tail, window_size=20, smooth_win=5, smooth_poly=2, multiplier=3, debug=False, padding=5, do_clip = False):


    
    """
    Compute the centerline of a self-overlapping 2D tube/worm polygon using sliding paired windows.

    Parameters
    ----------
    poly : Nx2 array
        Ordered polygon points of the worm outline.
    head : 2-array
        Coordinates of the head.
    tail : 2-array
        Coordinates of the tail.
    window_size : int
        Number of consecutive points from each half used to compute local midpoint candidates.
    smooth_win : int
        Window size for Savitzky-Golay smoothing of the midline.
    smooth_poly : int
        Polynomial order for Savitzky-Golay smoothing of the midline.
    plot : bool
        Plot the final outline and centerline.
    debug : bool
        Plot the two halves in different colors for debugging.
    padding : int
        Pixels to pad around polygon for area calculation.
    
    Returns
    -------
    midline : Mx2 array
        Smoothed centerline coordinates from head to tail.
    length : float
        Length of the centerline in pixels.
    area : int
        Area of the polygon in pixels.
    """
    def resample_fixed_n(coords, n_points):
        """Resample a 2D polyline to exactly n_points, evenly spaced along its arc-length."""
        coords = np.asarray(coords)
    
        diffs = np.diff(coords, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cumdist = np.hstack([0, np.cumsum(dists)])
    
        fx = interp1d(cumdist, coords[:,0], kind='linear')
        fy = interp1d(cumdist, coords[:,1], kind='linear')
    
        new_cumdist = np.linspace(0, cumdist[-1], n_points)
    
        return np.vstack([fx(new_cumdist), fy(new_cumdist)]).T
    import numpy as np
    from scipy.spatial.distance import cdist
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    from skimage.draw import polygon as sk_polygon

    poly = np.asarray(poly)
    head = poly[idx_head]
    tail = poly[idx_tail]

    # --- split polygon into two halves, both from head to tail ---
    if idx_head < idx_tail:
        half1 = poly[idx_head:idx_tail+1]
        half2 = np.vstack([poly[idx_tail:], poly[:idx_head+1]])
    else:
        half1 = np.vstack([poly[idx_head:], poly[:idx_tail+1]])
        half2 = poly[idx_tail:idx_head+1][::-1]

    # --- ensure both halves run head -> tail ---
    if np.linalg.norm(half1[0] - head) > np.linalg.norm(half1[-1] - head):
        half1 = half1[::-1]
    if np.linalg.norm(half2[0] - head) > np.linalg.norm(half2[-1] - head):
        half2 = half2[::-1]

    # --- use shorter half as template strand ---
    if len(half1) > len(half2):
        half1, half2 = half2, half1   # half1 = shorter strand
    
    # --- FORCE BOTH halves to have identical number of points ---
    n_points = len(half1)
    half1 = resample_fixed_n(half1, n_points)
    half2 = resample_fixed_n(half2, n_points)

    # --- compute midline with sliding window, local contralateral distance ---
    n_half1 = len(half1)
    n_half2 = len(half2)
    midline_points = []
    

    if debug: 
        fig,ax = plt.subplots()
        for i in range(half1.shape[0]-1):
            ax.plot(half1[i:i+1,0],half1[i:i+1,0])
    
    for i in range(n_half1 - window_size + 1):
        idx1_window = np.arange(i, i + window_size)
        # contralateral candidate indices ± window_size
        idx2_min = max(0, i - window_size*multiplier)
        idx2_max = min(n_half2, i + window_size + window_size*multiplier)
        # local distance matrix
        local_dmat = cdist(half1[idx1_window], half2[idx2_min:idx2_max])
        idx2_local = np.argmin(local_dmat, axis=1) + idx2_min
        # compute local midpoint
        local_mid = (half1[idx1_window] + half2[idx2_local]) / 2.0
        midline_points.append(local_mid[0])  # slide by 1

    # append last point to ensure coverage to tail
    midline_points.append((half1[-1] + half2[-1]) / 2.0)
    midline = np.array(midline_points)

    # --- clip to head and tail ---
    d_start = np.linalg.norm(midline - head, axis=1)
    d_end = np.linalg.norm(midline - tail, axis=1)
    start_idx = np.argmin(d_start)
    end_idx = np.argmin(d_end)
    if start_idx < end_idx:
        midline = midline[start_idx:end_idx+1]
    else:
        midline = midline[end_idx:start_idx+1][::-1]

    # --- prepend head and append tail ---
    midline = np.vstack([head, midline, tail])

    # --- smooth centerline ---
    if len(midline) >= smooth_win:
        midline[:,0] = savgol_filter(midline[:,0], smooth_win, smooth_poly)
        midline[:,1] = savgol_filter(midline[:,1], smooth_win, smooth_poly)


    diffs = np.diff(midline, axis=0)
    length = np.sum(np.linalg.norm(diffs, axis=1))

    # --- compute polygon area (rasterize for simplicity) ---
    min_xy = np.floor(poly.min(axis=0)) - padding
    poly_shifted = poly - min_xy + padding
    max_xy = np.ceil(poly_shifted.max(axis=0)) + padding
    img_shape = (int(max_xy[1])+1, int(max_xy[0])+1)
    rr, cc = sk_polygon(poly_shifted[:,1], poly_shifted[:,0], img_shape)
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask[rr, cc] = 1
    area = np.sum(mask)

    # --- optional plot ---
    if debug:
        plt.figure(figsize=(8,6))
        plt.plot(poly[:,0], poly[:,1], 'k-', label='Outline')
       
        plt.plot(half1[:,0], half1[:,1], 'r.-', label='Half 1 (template)')
        plt.plot(half2[:,0], half2[:,1], 'b.-', label='Half 2')
        plt.plot(midline[:,0], midline[:,1], 'k-', lw=2, label='Centerline')
        plt.scatter(head[0], head[1], c='g', s=50, label='Head')
        plt.scatter(tail[0], tail[1], c='m', s=50, label='Tail')
        plt.axis('equal')
        plt.legend()
        plt.show()

    return midline, length, area

def find_head_tail_robust(contour, win_frac=0.05, debug=False):
    """
    Robust head/tail detection for worm-like polygons.

    Parameters
    ----------
    contour : (N,2) array
        Closed worm outline
    dS : float
        Polygon simplification tolerance
    win_frac : float
        Neighborhood size as fraction of contour length
    debug : bool

    Returns
    -------
    head : (2,)
    tail : (2,)
    """

    # ---------------------------------------
    # Simplify & extract closed contour
    # ---------------------------------------
    #poly = Polygon(contour).simplify(dS)
    poly = Polygon(contour)
    pts = np.asarray(poly.exterior.coords[:-1])
    N = len(pts)

    if N < 40:
        raise ValueError("Contour too small for robust head/tail detection")

    # ---------------------------------------
    # Arc-length parameterization
    # ---------------------------------------
    dxy = np.diff(np.vstack([pts, pts[0]]), axis=0)
    ds = np.linalg.norm(dxy, axis=1)
    s = np.cumsum(ds)
    s = s / s[-1]  # normalize [0,1)

    # ---------------------------------------
    # Tangents & curvature (robust, closed)
    # ---------------------------------------
    edges = np.diff(np.vstack([pts, pts[0]]), axis=0)     # (N,2)
    ds = np.linalg.norm(edges, axis=1) + 1e-8             # (N,)
    
    angles = np.arctan2(edges[:, 1], edges[:, 0])         # (N,)
    dtheta = np.diff(np.unwrap(np.hstack([angles, angles[0]])))  # (N,)
    
    curvature = np.abs(dtheta) / ds                        # (N,)
    
    # ---------------------------------------
    # Smooth curvature with boundary extension
    # ---------------------------------------
    win = max(5, int(win_frac * N))
    if win % 2 == 0:
        win += 1
    if win >= N:
        win = N - 1 if (N - 1) % 2 else N - 2
    
    # extend signal by 5% to protect boundary peaks
    ext_frac = 0.05
    k_ext = max(3, int(ext_frac * N))
    
    curv_ext = np.concatenate([curvature, curvature[:k_ext]])
    
    curv_ext_s = scipy.signal.savgol_filter(
        curv_ext,
        window_length=win,
        polyorder=2,
        mode="interp"
    )
    
    # ---------------------------------------
    # Candidate ends = curvature peaks
    # ---------------------------------------
    # pks_ext, props = scipy.signal.find_peaks(
    #     curv_ext_s,
    #     prominence=np.percentile(curv_ext_s, 75),
    #     distance=win
    # )
    pks_ext, _ = scipy.signal.find_peaks(
    curv_ext_s,
    distance=win
    )
    
    # Fallback if too few peaks
    if len(pks_ext) < 2:
        # take the two largest values directly
        pks_ext = np.argsort(curv_ext_s)[-2:]
    else:
        # Sort peaks by height (descending) and keep top 2
        pks_ext = pks_ext[np.argsort(curv_ext_s[pks_ext])[-2:]]
        
    # map peaks back to original index range
    pks = np.unique(pks_ext % N)
    
    if len(pks) < 2:
        pks = np.argsort(curvature)[-4:]
    
    curvature_s = curv_ext_s[:N]  # keep only original segment

    # ---------------------------------------
    # Pick two most distant peaks
    # ---------------------------------------
    P = pts[pks]
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    i, j = np.unravel_index(np.argmax(D), D.shape)

    idx1, idx2 = pks[i], pks[j]

    # ---------------------------------------
    # Head vs tail decision
    # ---------------------------------------
    def end_score(idx):
        """Higher = more tail-like"""
        # local curvature
        curv = curvature_s[idx]

        # local symmetry (head more symmetric)
        k = win // 2
        left = curvature_s[(idx - k) % N]
        right = curvature_s[(idx + k) % N]
        asym = abs(left - right)

        return curv + 0.5 * asym

    s1 = end_score(idx1)
    s2 = end_score(idx2)

    if s1 > s2:
        tail_idx, head_idx = idx1, idx2
    else:
        tail_idx, head_idx = idx2, idx1

    head = pts[head_idx]
    tail = pts[tail_idx]

    # ---------------------------------------
    # Debug visualization
    # ---------------------------------------
    if debug:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].plot(curvature_s, lw=1)
        ax[0].scatter(pks, curvature_s[pks], c="k")
        ax[0].axvline(head_idx, color="g", label="head")
        ax[0].axvline(tail_idx, color="r", label="tail")
        ax[0].set_title("Curvature profile")
        ax[0].legend()

        ax[1].plot(pts[:, 0], pts[:, 1], "-k")
        ax[1].scatter(*head, c="g", s=80, label="head")
        ax[1].scatter(*tail, c="r", s=80, label="tail")
        ax[1].axis("equal")
        ax[1].legend()

        plt.show()

    return head, tail


def get_worm_centerline(poly, plot=False, padding=10, smooth_win=15, smooth_poly=3):
    """
    Extract centerline from a 2D polygon of a worm/tube, including self-overlapping parts.
    Uses skeleton graph + longest path along skeleton.
    """
    poly = np.asarray(poly)
    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    # Shift to positive coords
    min_xy = poly.min(axis=0)
    poly_shifted = poly - min_xy + padding

    # Rasterize polygon
    max_xy = np.ceil(poly_shifted.max(axis=0)) + padding
    img_shape = (int(max_xy[1])+1, int(max_xy[0])+1)
    rr, cc = sk_polygon(poly_shifted[:,1], poly_shifted[:,0], img_shape)
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask[rr, cc] = 1
    area = np.sum(mask)

    # Morphological close
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Skeletonize
    skeleton = skeletonize(mask > 0)

    # Build graph from skeleton
    ys, xs = np.where(skeleton)
    G = nx.Graph()
    for x, y in zip(xs, ys):
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x+dx, y+dy
                if 0 <= nx_ < mask.shape[1] and 0 <= ny_ < mask.shape[0]:
                    if skeleton[ny_, nx_]:
                        dist = np.sqrt(dx**2 + dy**2)
                        G.add_edge((x,y), (nx_,ny_), weight=dist)

    # Find all endpoints and branchpoints
    endpoints = [n for n, d in G.degree() if d == 1]
    branchpoints = [n for n, d in G.degree() if d >= 3]

    # Use all endpoints if available, else branchpoints
    if len(endpoints) >= 2:
        candidates = endpoints
    elif len(branchpoints) >= 2:
        candidates = branchpoints
    else:
        candidates = list(G.nodes)

    # Find longest path along skeleton (all pairs among candidates)
    longest_path = []
    longest_dist = -1
    for i, a in enumerate(candidates):
        for b in candidates[i+1:]:
            try:
                path = nx.shortest_path(G, source=a, target=b, weight='weight')
                dist = sum(np.linalg.norm(np.array(path[j])-np.array(path[j+1])) for j in range(len(path)-1))
                if dist > longest_dist:
                    longest_dist = dist
                    longest_path = path
            except nx.NetworkXNoPath:
                continue

    if not longest_path:
        raise ValueError("No centerline path found.")

    centerline_shifted = np.array(longest_path)[:, [0,1]].astype(float)
    centerline = centerline_shifted - padding + min_xy

    # Smooth
    if len(centerline) >= smooth_win:
        centerline[:,0] = savgol_filter(centerline[:,0], smooth_win, smooth_poly)
        centerline[:,1] = savgol_filter(centerline[:,1], smooth_win, smooth_poly)

    # Length
    diffs = np.diff(centerline, axis=0)
    length = np.sum(np.linalg.norm(diffs, axis=1))

    # Plot
    if plot:
        plt.figure(figsize=(8,6))
        plt.plot(poly[:,0], poly[:,1], 'b-', label='Outline')
        plt.plot(centerline[:,0], centerline[:,1], 'r-', lw=2, label='Centerline')
        plt.axis('equal')
        plt.legend()
        plt.title(f'Centerline length: {length:.2f}px')
        plt.show()

    return centerline, length, area




## werk: 
image_dir = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images' # this is where the images are sitting
point_csv =  '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_points.csv'
area_json =  '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_areas.json'
color_csv =  '/home/wormulon/models/worm_analytic_suite/class_colors.csv'





ids, bbs,segs,ls,ars = load_metrics_from_json(area_json)
a,loop_count,c = detect_loops_batch(bbs)
np.where(loop_count>0)


area_coords = np.array(segs[55][0])
area_coords = area_coords.reshape(-1,2)
area_coords1 = upsample_closed_contour(area_coords, N_min=400)
area_coords2 = smooth_polygon_gaussian(area_coords1, sigma=4.0, closed=True)

coarse_params = [
    {"num_start_points": 12},
    {"num_start_points": 6, "search_space": int(400/12)}
]

branch_params = [
    {"num_start_points": 5, "end_points_each": 5, "search_space": 15},
    {"num_start_points": 5, "end_points_each": 5, "search_space": 3}
]


metric_keys = [
    "length",
    "mean_smoothness",
    "peak_bend",
    "end_curvature_ratio",
    "inside_fraction"
]  # example

weights = {
    "length": 1.2,
    "mean_smoothness": -1.0,
    "peak_bend": 0.0,
    "end_curvature_ratio": 0,
    "inside_fraction": 1.0
}
 



output = centerline, best_candidate, candidate_db, mean_std = optimize_backbone_scaled(
    area_coords2,
    trace_centerline,
    coarse_params,
    branch_params,
    metric_keys,
    weights
)


# fig,ax = plt.subplots()
# ax.plot(area_coords[:,0],area_coords[:,1])
# plt.show()
# find_head_tail2(area_coords2, 1, 1, debug=True)
# find_head_tail_robust(area_coords2, dS=2.0, win_frac=0.05, debug=True)



cands,keys = candidate_db_to_array(output[2])
end_curvature_ratio = 0
head_idx = 1
peak_bend = 2
inside_fraction = 3
length = 4
tail_idx = 5
score = 6
mean_smoothness = 7
inside_all = 8
area = 9



idx = np.where(cands[:,score] == np.max(cands[:,score]))
print(cands[idx[0]][0,[length,inside_fraction,score]])
plot_candidate(
    area_coords2,
    output[2][int(idx[0])],
    ax=None,
    title=None,
    show_head_tail=True,
    lw_poly=1.5,
    lw_center=2.5
)

idx = np.where(cands[:,length] == np.max(cands[:,length]))
print(cands[idx[0]][0,[length,inside_fraction,score]])
plot_candidate(
    area_coords2,
    output[2][int(idx[0])],
    ax=None,
    title=None,
    show_head_tail=True,
    lw_poly=1.5,
    lw_center=2.5
)



idx = np.where(cands[:,length] == np.max(cands[:,length]))
print(cands[idx[0]][0,[length,end_curvature_ratio,inside_fraction,score]])
idx = np.where(cands[:,score] == np.max(cands[:,score]))
print(cands[idx[0]][0,[length,end_curvature_ratio,inside_fraction,score]])


fig,ax=plt.subplots()

plt.plot(cands[:,score])
plt.plot(cands[:,mean_smoothness])
plt.plot(cands[:,length])
plt.show

