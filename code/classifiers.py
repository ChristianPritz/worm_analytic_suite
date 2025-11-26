#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:06:31 2025

@author: wormulon
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
import json,copy,joblib
from matplotlib.patches import Polygon


def angle_histogram(coords, bins=20):
    """
    coords : (n, 2) array of xy coordinates
    bins   : number of histogram bins (default 20)

    Returns:
        hist_normalized : (bins,) array with normalized histogram counts
    """

    coords = np.asarray(coords)
    n = coords.shape[0]

    # Compute all pairwise difference vectors
    diff = coords[:, None, :] - coords[None, :, :]   # shape (n, n, 2)

    # Remove zero-length vectors (diagonal)
    mask = ~np.eye(n, dtype=bool)
    dx = diff[:,:,0][mask]
    dy = diff[:,:,1][mask]

    # Compute angles between 0 and 2*pi
    angles = np.arctan2(dy, dx)         # returns -pi .. pi
    angles = np.mod(angles, 2*np.pi)    # -> 0 .. 2*pi

    # Histogram
    hist, _ = np.histogram(angles, bins=bins, range=(0, 2*np.pi))

    # Normalize
    hist_normalized = hist / hist.sum() if hist.sum() > 0 else hist

    return hist_normalized


def normalize_zscore(X):
    """
    X: (n_samples, n_features) matrix

    Returns:
        X_normalized: same shape, standardized columns
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8  # small epsilon to avoid div by 0
    X_norm = (X - mean) / std
    return X_norm



def label_coco_areas(json_path):
    """
    Quick labeling of COCO area annotations.

    Parameters
    ----------
    json_path : str
        Path to COCO JSON file.

    Returns
    -------
    labels : list of int
        User-assigned labels for each annotation.
    """

    # --- load JSON ---
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # Map image_id to image size
    image_info = {img['id']: img for img in coco['images']}

    labels = []

    # --- iterate over annotations ---
    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_info = image_info[img_id]
        width, height = img_info['width'], img_info['height']

        # Area annotation: polygon (segmentation)
        segs = ann.get('segmentation', [])
        if len(segs) == 0:
            print(f"Annotation {ann['id']} has no segmentation, skipping.")
            labels.append(0)
            continue

        # COCO format: segmentation can be a list of lists (multiple polygons)
        # We'll take the first polygon
        polygon = segs[0]
        xy = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]

        # --- plot polygon ---
        plt.figure(figsize=(6,6))
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.gca().invert_yaxis()  # COCO y=0 is top
        plt.gca().set_aspect('equal')
        plt.gca().add_patch(Polygon(xy, closed=True, fill=True, color='orange', alpha=0.5))
        plt.title(f"Annotation ID: {ann['id']} - Image ID: {img_id}")
        plt.show()

        # --- prompt user ---
        user_input = input("Enter label (int), or press Enter for 0: ")
        if user_input.strip() == '':
            labels.append(0)
        else:
            try:
                labels.append(int(user_input))
            except ValueError:
                print("Invalid input, storing 0.")
                labels.append(0)

        plt.close()

    return labels

def coco_areas_calculation(json_path):
    """
    Quick labeling of COCO area annotations.

    Parameters
    ----------
    json_path : str
        Path to COCO JSON file.

    Returns
    -------
    labels : list of int
        User-assigned labels for each annotation.
    """

    # --- load JSON ---
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # Map image_id to image size
    image_info = {img['id']: img for img in coco['images']}

    hists = []

    # --- iterate over annotations ---
    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_info = image_info[img_id]
        width, height = img_info['width'], img_info['height']

        # Area annotation: polygon (segmentation)
        segs = ann.get('segmentation', [])
        polygon = segs[0]
        xy = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        xy = np.array(xy)
        print(xy.shape)
        hists.append(angle_histogram(xy))

        

    return np.array(hists)



def train_classifier(X, y, n_estimators=100, random_state=0,normalize=False):
    """
    X: (n_samples, 20) feature matrix
    y: binary labels (0/1 or False/True)

    Returns:
        trained Random Forest model
    """
    if normalize:
        X = copy.copy(X)
        X = normalize_zscore(X)    
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X, y)
    joblib.dump(model, "RF_classifier.pkl")
    return model


def classify(X,model=None, threshold=0.5,normalize=False):
    """
    model: trained Random Forest model
    X: data to classify
    threshold: probability cutoff for 'yes'

    Returns:
        preds: binary predictions using the threshold
        probs: model probability for class '1'
    """
    if model is None:
        model = load_model()
        #model = joblib.load("RF_classifier.pkl")
    if normalize:
        X = copy.copy(X)
        X = normalize_zscore(X) 
    
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, probs


def run_kmeans_and_show(X, external_label, k):
    """
    Runs k-means, sorts the matrix by cluster labels, and displays:
      - heatmap of sorted X
      - external labels as aligned side bar

    Parameters
    ----------
    X : array (n_samples, n_features)
    external_label : array (n_samples,), numeric values
    k : number of clusters
    """

    # --- run k-means ---
    kmeans = KMeans(
        n_clusters=k,
        n_init="auto",
        random_state=0
    )
    labels = kmeans.fit_predict(X)

    # --- sort by cluster labels ---
    sort_idx = np.argsort(labels)
    X_sorted = X[sort_idx]
    sorted_labels = labels[sort_idx]
    external_sorted = np.asarray(external_label)[sort_idx]

    # --- plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), 
                             gridspec_kw={'width_ratios': [5, 1]})

    # Left: matrix
    im0 = axes[0].imshow(X_sorted, aspect='auto', interpolation='nearest')
    axes[0].set_title(f"K-means sorted matrix (k={k})")
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Samples (sorted by cluster)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Right: external label column
    # reshape to (n,1) so it displays as a vertical stripe
    im1 = axes[1].imshow(external_sorted.reshape(-1, 1), 
                         aspect='auto', interpolation='nearest')
    axes[1].set_title("External\nLabel")
    axes[1].set_xticks([])  # hide x ticks
    axes[1].set_yticks([])  # hide y ticks
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return labels, X_sorted, external_sorted, sorted_labels, kmeans

#-----------------------------------------------------------------------------#
#
# HELPER FUNCTIONS
#
#-----------------------------------------------------------------------------#

def load_model():
    import os, joblib
    path = os.path.join(os.path.dirname(__file__), "RF_classifier.pkl")
    return joblib.load(path)