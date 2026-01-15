#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 11:55:22 2026

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


    
def backbones_to_pca_matrix(backbones):
    """
    Convert (n, l, 2) backbone array to PCA-ready matrix.

    NaNs are handled by column-wise mean imputation.

    Parameters
    ----------
    backbones : np.ndarray (n, l, 2)

    Returns
    -------
    X : np.ndarray (n, l*2)
    """

    n, l, _ = backbones.shape
    X = backbones.reshape(n, l * 2)

    # Column-wise mean (ignoring NaNs)
    col_mean = np.nanmean(X, axis=0)

    # Replace NaNs with column mean
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    return X


def run_pca(X, n_components=None):
    """
    Run PCA on backbone matrix.

    Parameters
    ----------
    X : np.ndarray (n, features)
    n_components : int or None

    Returns
    -------
    pca : sklearn PCA object
    scores : np.ndarray (n, n_components)
    """

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    return pca, scores

def plot_variance_explained(pca):
    """
    Plot explained variance ratio.
    """

    var = pca.explained_variance_ratio_
    cumvar = np.cumsum(var)

    plt.figure(figsize=(6, 4))
    plt.plot(var * 100, 'o-', label="Individual")
    plt.plot(cumvar * 100, 's--', label="Cumulative")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained (%)")
    plt.title("PCA Variance Explained")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pca_scatter(scores, pc_x=1, pc_y=2, color=None, labels=None, colors=None):
    """
    Plot PCA scatter for user-selected PCs with optional labels and custom colors.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples, n_components)
        PCA scores
    pc_x : int
        PC for x-axis (1-based)
    pc_y : int
        PC for y-axis (1-based)
    color : array-like or None
        Optional continuous color values per sample (overrides labels if given)
    labels : array-like or None
        Optional categorical labels per sample (for coloring)
    colors : dict or list or None
        Optional mapping of label -> color, or list of colors to use
    """
    ix = pc_x - 1
    iy = pc_y - 1

    plt.figure(figsize=(6, 6))

    if labels is not None:
        labels = np.array(labels)
        unique_labels = np.unique(labels)

        # Generate colors if not provided
        if colors is None:
            rng = np.random.default_rng(42)
            cmap = plt.get_cmap("tab20")
            colors = {lab: cmap(i % 20) for i, lab in enumerate(unique_labels)}
        elif isinstance(colors, list):
            colors = {lab: colors[i % len(colors)] for i, lab in enumerate(unique_labels)}

        for lab in unique_labels:
            mask = labels == lab
            plt.scatter(scores[mask, ix], scores[mask, iy], 
                        c=[colors[lab]]*np.sum(mask), label=str(lab), s=40)
        plt.legend(title="Labels")

    else:
        # fallback to continuous color or single color
        sc = plt.scatter(scores[:, ix], scores[:, iy], c=color, cmap="viridis", s=40)
        if color is not None:
            plt.colorbar(sc, label="Color value")

    plt.xlabel(f"PC{pc_x}")
    plt.ylabel(f"PC{pc_y}")
    plt.title(f"PCA Scatter: PC{pc_x} vs PC{pc_y}")
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.tight_layout()
    plt.show()


def seg_2_arr(seg):
    seg = np.array(seg)
    return seg.reshape((int(seg.shape[1]/2),2))

def compute_circumferences(segs):
    def compute_circumference(seg):
        seg = np.vstack((seg,seg[0,:]))
        diffs = np.diff(seg, axis=0)
        return np.nansum(np.linalg.norm(diffs, axis=1))
    if isinstance(segs, np.ndarray):
        return compute_circumference(seg)
    else:
        circums = []
        for i in segs:
            circums.append(compute_circumference(seg_2_arr(i)))
        return circums
                          
def label_arrays(arrays):
    """
    Plot each 2D numpy array and ask user for a binary label.

    Parameters
    ----------
    arrays : list of np.ndarray (2D)

    Returns
    -------
    labels : np.ndarray (n,)
        User labels (0 or 1)
    """

    labels = []

    for i, arr in enumerate(arrays):
        arr = seg_2_arr(arr)
        plt.figure()
        plt.plot(arr[:, 0], arr[:, 1], '-o')
        plt.title(f"Sample {i}")
        plt.axis('equal')
        plt.show(block=False)

        user_input = input("Enter 1 or 0 (empty = 0): ").strip()

        if user_input == "1":
            labels.append(1)
        else:
            labels.append(0)

        plt.close()

    return np.array(labels)    


def train_outlier_svm(X, y, test_size=0.2, kernel="rbf", C=1.0, gamma="scale"):
    """
    Train a binary SVM to separate outliers from inliers.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Feature matrix
    y : np.ndarray (n_samples,)
        Labels (0=inlier, 1=outlier)
    test_size : float
        Fraction for validation split
    kernel : str
        SVM kernel ('linear', 'rbf', 'poly', 'sigmoid')
    C : float
        SVM regularization
    gamma : str or float
        Kernel coefficient for 'rbf', 'poly', 'sigmoid'

    Returns
    -------
    model : trained SVC
    scaler : fitted StandardScaler
    X_val, y_val : validation set for evaluation
    """

    # Standardize features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Split for evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        Xs, y, test_size=test_size, random_state=42, stratify=y
    )

    # Train SVM
    model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced")
    model.fit(X_train, y_train)

    return model, scaler, X_val, y_val

def evaluate_svm(model, scaler, X_val, y_val):
    X_val_s = scaler.transform(X_val)
    y_pred = model.predict(X_val_s)
    
    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))
    
    return y_pred

def plot_ocsvm_2d(X, labels, model, scaler):
    Xs = scaler.transform(X)

    xx, yy = np.meshgrid(
        np.linspace(Xs[:,0].min()-1, Xs[:,0].max()+1, 200),
        np.linspace(Xs[:,1].min()-1, Xs[:,1].max()+1, 200)
    )
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z < 0, alpha=0.3)
    plt.scatter(Xs[:,0], Xs[:,1], c=(labels == -1), cmap='coolwarm', s=30)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("One-Class SVM Outlier Detection")
    plt.show()

def compute_local_curvature(backbone):
    """
    Compute local curvature along a 2D backbone.

    Parameters
    ----------
    backbone : np.ndarray, shape (L,2)
        Nx2 array of (x,y) coordinates along the centerline

    Returns
    -------
    curvature : np.ndarray, shape (L,)
        Curvature at each point. Endpoints are set to 0.
    """

    if backbone.shape[0] < 3:
        return np.zeros(backbone.shape[0])

    x = backbone[:, 0]
    y = backbone[:, 1]

    # First derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula
    # kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = numerator / denominator
        kappa[np.isnan(kappa)] = 0  # handle division by zero

    return kappa

def backbone_curvature_stats(backbones):
    """
    Compute local curvature statistics for multiple backbones.

    Parameters
    ----------
    backbones : np.ndarray, shape (n, l, 2)
        Array of n backbones, each with l points (x, y).

    Returns
    -------
    curvature_stats : dict of np.ndarray, shape (n,)
        'min', 'max', 'mean' curvature for each backbone
    """
    if isinstance(backbones,list):
        n = len(backbones)
    else:
        n = backbones.shape[0]

    curv_min = np.zeros(n)
    curv_max = np.zeros(n)
    curv_mean = np.zeros(n)
    curv_std = np.zeros(n)
    curv_q95 = np.zeros(n)
    curv_q05 = np.zeros(n)
    
    for i in range(n):
        bb = backbones[i]
        if isinstance(bb, list):
            bb = np.array(bb[0])
            bb = bb.reshape(-1,2)
        
        # skip empty backbones
        if np.isnan(bb).all() or bb.shape[0] < 3:
            curv_min[i] = np.nan
            curv_max[i] = np.nan
            curv_mean[i] = np.nan
            continue

        # compute local curvature
        x = bb[:,0]
        y = bb[:,1]

        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        numerator = dx * ddy - dy * ddx
        denominator = (dx**2 + dy**2)**1.5
        with np.errstate(divide='ignore', invalid='ignore'):
            kappa = numerator / denominator
            kappa[np.isnan(kappa)] = 0

        curv_min[i] = np.min(kappa)
        curv_max[i] = np.max(kappa)
        curv_mean[i] = np.mean(kappa)
        curv_std[i] = np.std(kappa)
        curv_q95[i] = np.quantile(kappa,0.95)
        curv_q05[i] = np.quantile(kappa,0.05)

    curvature_stats = np.hstack((curv_min.reshape((curv_min.size,1)),
                                 curv_max.reshape((curv_min.size,1)),
                                 curv_mean.reshape((curv_min.size,1)),
                                 curv_std.reshape((curv_min.size,1)),
                                 curv_q95.reshape((curv_min.size,1)),
                                 curv_q05.reshape((curv_min.size,1))))

    return curvature_stats

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


def compute_polygon_bulbiness_batch(polygons, backbones):
    """
    Compute a bulbiness metric for each polygon in a batch.
    
    Bulbiness is defined as the ratio of the polygon area to the area of a
    uniform tube approximated along the backbone:
        Bulbiness = Polygon Area / (Backbone Length * Mean Width)
    
    Parameters
    ----------
    polygons : list of np.ndarray
        Each element is an Nx2 array representing polygon vertices.
    backbones : list of np.ndarray
        Each element is an Mx2 array representing the backbone corresponding to the polygon.
    
    Returns
    -------
    bulbiness : np.ndarray, shape (n,)
        Bulbiness metric for each polygon. NaN if invalid input.
    """
    n = len(polygons)
    bulbiness = np.full(n, np.nan, dtype=float)

    for i in range(n):
        poly_pts = np.array(polygons[i][0])
        poly_pts = poly_pts.reshape(-1,2)
        bb = backbones[i]

        if poly_pts is None or bb is None or len(poly_pts) < 3 or len(bb) < 2:
            continue

        try:
            poly = Polygon(poly_pts)
            if not poly.is_valid or poly.area == 0:
                continue

            # approximate local width along backbone
            line = LineString(bb)
            # distances from backbone points to polygon boundary
            distances = []
            for pt in bb:
                p = Point(pt)
                nearest = nearest_points(p, poly.exterior)[1]
                d = np.linalg.norm(np.array(pt) - np.array(nearest.coords[0]))
                distances.append(d)
            distances = np.array(distances)
            mean_width = np.mean(distances) * 2  # width = 2 * distance to edge

            # backbone length
            bb_diff = np.diff(bb, axis=0)
            bb_length = np.sum(np.sqrt(np.sum(bb_diff**2, axis=1)))

            # bulbiness
            bulbiness[i] = poly.area / (bb_length * mean_width)

        except Exception as e:
            bulbiness[i] = np.nan
            print(f"Warning: polygon {i} failed: {e}")

    return bulbiness



def polygon_area(coords):
    """
    Compute polygon area using the shoelace formula.
    coords: (n,2) ordered polygon
    """
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


def concavity_convexity(coords):
    """
    Compute single-number concavity and convexity metrics for a shape.

    Parameters
    ----------
    coords : np.ndarray (n, 2)
        Ordered polygon coordinates

    Returns
    -------
    convexity : float
        Area(shape) / Area(convex hull), in [0, 1]
    concavity : float
        (Area(hull) - Area(shape)) / Area(hull), in [0, 1)
    """
    coords = np.asarray(coords)

    if coords.shape[0] < 3:
        return np.nan, np.nan

    area_shape = polygon_area(coords)

    hull = ConvexHull(coords)
    hull_coords = coords[hull.vertices]
    area_hull = polygon_area(hull_coords)

    if area_hull == 0:
        return np.nan, np.nan

    convexity = area_shape / area_hull
    concavity = (area_hull - area_shape) / area_hull

    return convexity, concavity

def batch_concavity_convexity(shapes):
    """
    Compute concavity and convexity metrics for a list of polygon shapes.

    Parameters
    ----------
    shapes : list of np.ndarray
        Each element is an (n_i, 2) array of ordered polygon coordinates

    Returns
    -------
    convexity : np.ndarray (n_shapes,)
        Convexity values per shape
    concavity : np.ndarray (n_shapes,)
        Concavity values per shape
    """
    n = len(shapes)
    convexity = np.full(n, np.nan)
    concavity = np.full(n, np.nan)
    print('asdfasd')
    for i, coords in enumerate(shapes):
        coords = np.array(coords[0])
        print(coords)
        coords = coords.reshape(-1,2)
        
        try:
            ccvx, ccv = concavity_convexity(coords)
            convexity[i] = ccvx
            concavity[i] = ccv
        except Exception:
            # robust against degenerate shapes
            continue

    return convexity, concavity



## werk: 
image_dir = '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/images' # this is where the images are sitting
point_csv =  '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_points.csv'
area_json =  '/media/my_device/space worms/makesenseai_analyzed_images/to analyze/output/annotations_areas.json'
color_csv =  '/home/wormulon/models/worm_analytic_suite/class_colors.csv'






ids, bbs,segs,ls,ars = load_metrics_from_json(area_json)
circs = np.array(compute_circumferences(segs))
labels = label_arrays(segs)
kappas = backbone_curvature_stats(bbs)
kappas2 = backbone_curvature_stats(segs)
a,loop_count,c = detect_loops_batch(bbs)

compute_polygon_bulbiness_batch(segs,bbs)
bulbiness = compute_polygon_bulbiness_batch(segs,bbs)
conc,conv = batch_concavity_convexity(segs)




data = np.hstack((ls.reshape((ls.size,1)),
               ars.reshape((ars.size,1)),
               circs.reshape((circs.size,1)),
               np.square(circs.reshape((circs.size,1))),
               loop_count.reshape((loop_count.size,1)),
               conv.reshape((conv.size,1)),
               conc.reshape((conc.size,1)),
               kappas2,kappas))

X = scipy.stats.zscore(data,axis=0)
pca, scores = run_pca(X)
plot_variance_explained(pca)
for i in range(1,X.shape[1]):
    print(i)
    plot_pca_scatter(scores, pc_x=i-1, pc_y=i,labels=labels)



##Backbone problem.............................................................
import numpy as np
import scipy.signal
import scipy.ndimage

def find_head_tail_robust(contour, smooth_sigma=2.0, snap_window=15, debug=False):
    """
    Robust head/tail detection from a worm outline.
    Handles straight/blunt ends by snapping endpoints to true extremities.
    """

    c = np.asarray(contour)
    N = len(c)

    # --- Smooth contour ---
    xs = scipy.ndimage.gaussian_filter1d(c[:, 0], smooth_sigma, mode="wrap")
    ys = scipy.ndimage.gaussian_filter1d(c[:, 1], smooth_sigma, mode="wrap")
    c_s = np.column_stack((xs, ys))

    # --- Index distance (geodesic proxy) ---
    idx = np.arange(N)
    D = np.abs(idx[:, None] - idx[None, :])
    D = np.minimum(D, N - D)

    # --- Geodesic extrema ---
    i, j = np.unravel_index(np.argmax(D), D.shape)

    # --- Tangents ---
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    tangents = np.column_stack((dx, dy))
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8

    # --- Snap endpoint to true extremity ---
    def snap_endpoint(idx0):
        w = snap_window
        inds = (idx0 + np.arange(-w, w + 1)) % N
        base = c_s[idx0]
        t = tangents[idx0]

        proj = (c_s[inds] - base) @ t
        return inds[np.argmax(np.abs(proj))]

    i_s = snap_endpoint(i)
    j_s = snap_endpoint(j)

    p1 = c_s[i_s]
    p2 = c_s[j_s]

    # --- Curvature (ONLY for labeling) ---
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    curvature = np.nan_to_num(curvature)

    # --- Head vs tail assignment ---
    if curvature[i_s] > curvature[j_s]:
        head, tail = p1, p2
    else:
        head, tail = p2, p1

    # --- Debug ---
    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5,5))
        plt.plot(c_s[:,0], c_s[:,1], '-k')
        plt.scatter(*head, c='r', label='head')
        plt.scatter(*tail, c='b', label='tail')
        plt.scatter(c_s[[i, j],0], c_s[[i, j],1], c='y', label='raw extrema')
        plt.axis('equal')
        plt.legend()
        plt.title("Head/Tail with extremity snapping")
        plt.show()

    return head, tail


import numpy as np
from scipy.interpolate import interp1d

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



import numpy as np
import scipy.stats
import scipy.signal
import scipy.ndimage
from shapely.geometry import Polygon
import matplotlib.pyplot as pl


def my_find_head_tail(c, dS=2.0, bloc_int=5, debug=False):
    """
    Robust head/tail detection from a worm outline polygon.

    Parameters
    ----------
    c : (N,2) ndarray
        Closed contour (polygon)
    dS : float
        Polygon simplification tolerance
    bloc_int : int
        Neighborhood size for local curvature
    debug : bool

    Returns
    -------
    head : np.ndarray (2,)
    tail : np.ndarray (2,)
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def wrap_pi(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def cyclic_blocks(values, k):
        """Return stacked cyclic neighborhood blocks"""
        n = len(values)
        idx = np.arange(n)
        blocks = np.empty((2 * k + 1, n))
        blocks[k] = values
        for i in range(1, k + 1):
            blocks[k - i] = values[(idx - i) % n]
            blocks[k + i] = values[(idx + i) % n]
        return blocks

    def circular_variance(angles):
        C = np.mean(np.cos(angles), axis=0)
        S = np.mean(np.sin(angles), axis=0)
        return 1 - np.sqrt(C**2 + S**2)

    # ------------------------------------------------------------------
    # Simplify & extract polygon
    # ------------------------------------------------------------------
    poly = Polygon(c).simplify(dS)
    x, y = np.asarray(poly.exterior.coords.xy)
    xy = np.column_stack((x[:-1], y[:-1]))  # remove duplicate last point
    N = len(xy)

    # ------------------------------------------------------------------
    # Arc-length & geometry
    # ------------------------------------------------------------------
    dx = np.diff(xy[:, 0], append=xy[0, 0])
    dy = np.diff(xy[:, 1], append=xy[0, 1])
    seglen = np.sqrt(dx**2 + dy**2)

    angles = wrap_pi(np.arctan2(dy, dx))

    # ------------------------------------------------------------------
    # Local curvature proxy (robust)
    # ------------------------------------------------------------------
    ang_blocks = cyclic_blocks(angles, bloc_int)
    curv = circular_variance(ang_blocks)

    # Smooth curvature slightly
    curv = scipy.ndimage.gaussian_filter1d(curv, 1.0, mode="wrap")

    # ------------------------------------------------------------------
    # Candidate peaks
    # ------------------------------------------------------------------
    peaks, _ = scipy.signal.find_peaks(curv, distance=bloc_int)
    if len(peaks) < 2:
        raise RuntimeError("Not enough curvature peaks detected")

    # Keep strongest K peaks
    K = min(6, len(peaks))
    top = peaks[np.argsort(curv[peaks])[-K:]]

    # ------------------------------------------------------------------
    # Pairwise scoring
    # ------------------------------------------------------------------
    centroid = xy.mean(axis=0)
    dist_centroid = np.linalg.norm(xy - centroid, axis=1)

    best_score = -np.inf
    best_pair = None

    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            p, q = top[i], top[j]

            # Euclidean separation
            d_euc = np.linalg.norm(xy[p] - xy[q])

            # Arc-length separation (cyclic)
            d_arc = abs(p - q)
            d_arc = min(d_arc, N - d_arc)

            # Curvature strength
            c_score = curv[p] + curv[q]

            # Extremity prior
            e_score = dist_centroid[p] + dist_centroid[q]

            score = (
                1.0 * d_euc +
                0.5 * d_arc +
                1.0 * c_score +
                0.5 * e_score
            )

            if score > best_score:
                best_score = score
                best_pair = (p, q)

    i, j = best_pair

    # ------------------------------------------------------------------
    # Head vs tail assignment
    # (tail slightly more curved on average)
    # ------------------------------------------------------------------
    if curv[i] > curv[j]:
        tail_idx, head_idx = i, j
    else:
        tail_idx, head_idx = j, i

    head = xy[head_idx]
    tail = xy[tail_idx]

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    if debug:
        plt.figure(figsize=(5, 5))
        plt.plot(xy[:, 0], xy[:, 1], "-k")
        plt.scatter(xy[top, 0], xy[top, 1], c="orange", label="candidates")
        plt.scatter(*head, c="r", s=60, label="head")
        plt.scatter(*tail, c="b", s=60, label="tail")
        plt.axis("equal")
        plt.legend()
        plt.title("Head / Tail detection")
        plt.show()

        plt.figure()
        plt.plot(curv)
        plt.scatter(top, curv[top])
        plt.title("Curvature proxy")
        plt.show()

    return head, tail

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

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import triangle as tr
from scipy.sparse import lil_matrix, csgraph
from scipy.sparse.linalg import eigsh
from scipy.interpolate import splprep, splev

def harmonic_centerline(polygon_points, n_points=100, debug=False):
    """
    Compute a smooth backbone using harmonic/Laplacian centerline from a polygon.

    Parameters
    ----------
    polygon_points : np.ndarray (N,2)
        Closed polygon (x,y)
    n_points : int
        Number of points to sample along the backbone
    debug : bool
        Show diagnostic plots

    Returns
    -------
    backbone : np.ndarray (n_points,2)
        Smooth backbone coordinates
    """

    # --- Step 1: Triangulate the polygon ---
    poly = Polygon(polygon_points)
    if not poly.is_valid:
        poly = poly.buffer(0)  # fix self-intersections

    coords = np.array(poly.exterior.coords)
    segments = np.array([[i, i+1] for i in range(len(coords)-1)])
    tri_dict = dict(vertices=coords[:, :2], segments=segments)
    t = tr.triangulate(tri_dict, 'p')  # 'p' = PSLG

    vertices = t['vertices']
    triangles = t['triangles']

    # --- Step 2: Build Laplacian ---
    nV = len(vertices)
    L = lil_matrix((nV, nV))
    for tri in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    L[tri[i], tri[j]] = -1
                    L[tri[i], tri[i]] += 1

    # --- Step 3: Solve eigenvector (Fiedler vector) ---
    # L * f = lambda * f
    # Use smallest nonzero eigenvector
    vals, vecs = eigsh(L.tocsr(), k=2, which='SM')
    fiedler = vecs[:, 1]  # second smallest

    # --- Step 4: Rasterize vertices along Fiedler values ---
    idx_sorted = np.argsort(fiedler)
    backbone_pts = vertices[idx_sorted]

    # --- Step 5: Fit spline for smoothness ---
    tck, u = splprep([backbone_pts[:,0], backbone_pts[:,1]], s=0)
    u_new = np.linspace(0,1,n_points)
    x_new, y_new = splev(u_new, tck)
    backbone = np.column_stack([x_new, y_new])

    # --- Debug plots ---
    if debug:
        plt.figure(figsize=(6,6))
        plt.plot(coords[:,0], coords[:,1], '-k', label='polygon')
        plt.plot(backbone[:,0], backbone[:,1], '-r', lw=2, label='centerline')
        plt.scatter(vertices[:,0], vertices[:,1], c=fiedler, cmap='viridis', s=20)
        plt.axis('equal')
        plt.legend()
        plt.title('Harmonic / Laplacian Centerline')
        plt.show()

    return backbone

import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import spsolve
from shapely.geometry import Polygon
from scipy.interpolate import interp1d

def laplacian_centerline(polygon, head_idx=None, tail_idx=None, n_points=100):
    """
    Compute a Laplacian/harmonic centerline from a polygon.
    
    Parameters
    ----------
    polygon : np.ndarray (N,2)
        Closed polygon of the worm outline
    head_idx : int, optional
        Index of approximate head vertex
    tail_idx : int, optional
        Index of approximate tail vertex
    n_points : int
        Number of points to resample backbone to
    
    Returns
    -------
    backbone : np.ndarray (n_points, 2)
    """
    c = np.asarray(polygon)
    N = len(c)
    
    # --- Ensure polygon is closed ---
    if not np.all(c[0] == c[-1]):
        c = np.vstack([c, c[0]])
    
    # --- Graph adjacency: connect neighbors ---
    W = np.zeros((N, N))
    for i in range(N):
        W[i, (i-1)%N] = np.linalg.norm(c[i] - c[(i-1)%N])
        W[i, (i+1)%N] = np.linalg.norm(c[i] - c[(i+1)%N])
    
    # --- Graph Laplacian ---
    L = csgraph.laplacian(W, normed=False)
    
    # --- Set boundary vertices (Dirichlet) ---
    if head_idx is None:
        head_idx = 0
    if tail_idx is None:
        tail_idx = N//2
    
    boundary = np.zeros(N, dtype=bool)
    boundary[[head_idx, tail_idx]] = True
    
    # --- Solve harmonic coordinates ---
    coords = np.zeros_like(c, dtype=float)
    for dim in range(2):
        b = np.zeros(N)
        b[head_idx] = c[head_idx, dim]
        b[tail_idx] = c[tail_idx, dim]
        
        # Set up system: L * x = 0, with boundary fixed
        L_mod = L.copy()
        for idx in np.where(boundary)[0]:
            L_mod[idx, :] = 0
            L_mod[idx, idx] = 1
        x = spsolve(L_mod, b)
        coords[:, dim] = x
    
    # --- Trace backbone along harmonic coordinate (linear interpolation between head/tail) ---
    # Simple: order vertices by harmonic coordinate from head to tail
    h_val = coords[head_idx, 0] + coords[head_idx,1]
    t_val = coords[tail_idx,0] + coords[tail_idx,1]
    
    path_order = np.argsort(np.sum(coords, axis=1))
    backbone = c[path_order]
    
    # --- Resample to fixed number of points ---
    s = np.cumsum(np.linalg.norm(np.diff(backbone, axis=0), axis=1))
    s = np.insert(s, 0, 0)
    f = interp1d(s, backbone, axis=0)
    s_new = np.linspace(0, s[-1], n_points)
    backbone_resampled = f(s_new)
    
    if 10>1:
        plt.figure(figsize=(6,6))
        plt.plot(coords[:,0], coords[:,1], '-k', label='polygon')
        plt.plot(backbone_resampled[:,0], backbone_resampled[:,1], '-r', lw=2, label='centerline')
        plt.axis('equal')
        plt.legend()
        plt.title('Harmonic / Laplacian Centerline')
        plt.show()
    
    return backbone_resampled




for idx,i in enumerate(segs):
    print(idx)
    area_coords = np.array(i[0])
    area_coords = area_coords.reshape(-1,2)
    # if area_coords.shape[0] < 100:
    #     area_coords = upsample_closed_contour(area_coords, N_min=400)
    #     area_coords2 = smooth_polygon_gaussian(area_coords, sigma=2.0, closed=True)
    #     fig,ax = plt.subplots()
    #     ax.plot(area_coords[:,0],area_coords[:,1])
    #     ax.set_title("My plot title")
    #     plt.show
    # else:
    #     area_coords2 = upsample_closed_contour(area_coords, N_min=300)    
    area_coords1 = upsample_closed_contour(area_coords, N_min=400)
    area_coords2 = smooth_polygon_gaussian(area_coords1, sigma=4.0, closed=True)    
    laplacian_centerline(area_coords2, head_idx=None, tail_idx=None, n_points=100)




for idx,i in enumerate(segs):
    print(idx)
    area_coords = np.array(i[0])
    area_coords = area_coords.reshape(-1,2)
    # if area_coords.shape[0] < 100:
    #     area_coords = upsample_closed_contour(area_coords, N_min=400)
    #     area_coords2 = smooth_polygon_gaussian(area_coords, sigma=2.0, closed=True)
    #     fig,ax = plt.subplots()
    #     ax.plot(area_coords[:,0],area_coords[:,1])
    #     ax.set_title("My plot title")
    #     plt.show
    # else:
    #     area_coords2 = upsample_closed_contour(area_coords, N_min=300)    
    area_coords1 = upsample_closed_contour(area_coords, N_min=400)
    area_coords2 = smooth_polygon_gaussian(area_coords1, sigma=4.0, closed=True)
        
    #h,t = find_head_tail_robust(area_coords, smooth_sigma=.5, debug = True)
    h,t = find_head_tail(area_coords2, dS=2.0, bloc_int=10, debug = True)
    centerline,_,area = get_worm_centerline_sliding(area_coords2, h, t, window_size=5, multiplier=1, smooth_win=15, smooth_poly=3, debug=True)
   
    #selfoverlap = check_self_overlap(area_coords)
    #centerline,t,area = get_worm_centerline(area_coords, plot=False, padding=10)
    #if loop_count[idx]>0:
    #    h,t = find_head_tail(area_coords, 3, 5, debug=True)
    #    centerline,_,area = get_worm_centerline_sliding(area_coords, h, t, window_size=5, multiplier=1, smooth_win=15, smooth_poly=3, debug=True)








colors = np.hstack((np.ones((labels.shape[0],1)),np.zeros((labels.shape[0],1))))
colors = np.hstack((colors,labels.reshape((labels.shape[0],1))))
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ars,circs,ls,color=colors)
plt.show()


#TRAIN AN SVM


X = np.hstack((ls.reshape((ls.size,1)),ars.reshape((ars.size,1)),np.array(circs).reshape((len(circs),1))))
train_oneclass_svm(X,)

model, scaler, X_val, y_val = train_outlier_svm(X,labels,test_size=0.5)
preds = evaluate_svm(model, scaler, X, labels)

colors = np.hstack((np.ones((preds.shape[0],1)),np.zeros((preds.shape[0],1))))
colors = np.hstack((colors,preds.reshape((preds.shape[0],1))))
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ars,circs,ls,color=colors)
plt.show()



fig,ax = plt.subplots(dpi=600)
ax.scatter(ls,circs)
plt.show()
fig,ax = plt.subplots(dpi=600)
ax.scatter(ars,circs)
plt.show()
fig,ax = plt.subplots(dpi=600)
ax.scatter(ls,ars)
plt.show()



# check 

idx = np.where(np.multiply(ls<100, np.array(circs)<1000))


fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ars,circs,ls)
plt.show()


fig,ax = plt.subplots(dpi=277)
seg = seg_2_arr(segs[276])
ax.plot(seg[:,0],seg[:,1])
plt.show()



bbs_norm = normalize_backbones(bbs)

fig,ax = plt.subplots()
ax.plot(bbs[0,:,0],bbs[0,:,1])
ax.plot(bbs[1,:,0],bbs[1,:,1])
plt.show


fig,ax = plt.subplots()
ax.plot(bbs_norm[0,:,0],bbs_norm[0,:,1])
ax.plot(bbs_norm[1,:,0],bbs_norm[1,:,1])
ax.plot(bbs_norm[2,:,0],bbs_norm[3,:,1])
ax.plot(bbs_norm[100,:,0],bbs_norm[100,:,1])
ax.plot(bbs_norm[200,:,0],bbs_norm[200,:,1])

plt.show

killDx = np.where(np.sum(np.isnan(bbs_norm),axis=1)[:,0] == bbs_norm.shape[1]-2)
bbs_full = np.delete(bbs_norm,killDx,0)



n,r,c = bbs_full.shape
X_r = bbs_full.reshape(n,r*c)
X = scipy.stats.zscore(X_r,axis=0)
X = np.delete(X,[0,1],axis=1) #first column all 0s no info --> zscore nan

fig,ax = plt.subplots()
ax.imshow(np.isnan(X))
plt.show()








