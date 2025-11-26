#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:50:22 2025

@author: christian
"""

import numpy as np
import random
from shapely.geometry import Point
from shapely.affinity import scale
from shapely.ops import unary_union
from measurements import rim_score
import matplotlib.pyplot as plt

def generate_worm_polygon(center, length=300, width=40, num_vertices=40, curvature=0.2):
    """
    Generate an oblong worm-shaped polygon around a center point with random rotation.

    Parameters
    ----------
    center : tuple
        (x, y) center of the polygon
    length : float
        Approximate length of the worm
    width : float
        Approximate width of the worm
    num_vertices : int
        Number of vertices around the perimeter
    curvature : float
        Max vertical deviation along the worm axis (as fraction of width)

    Returns
    -------
    np.ndarray of shape (num_vertices, 2)
    """
    # Generate evenly spaced points along the long axis
    t = np.linspace(-0.5, 0.5, num_vertices//2)
    # Upper side
    x_upper = t * length
    y_upper = (width/2) * (1 + curvature*(np.random.rand(len(t))*2-1))
    # Lower side
    x_lower = t[::-1] * length
    y_lower = - (width/2) * (1 + curvature*(np.random.rand(len(t))*2-1))
    # Combine
    x = np.concatenate([x_upper, x_lower])
    y = np.concatenate([y_upper, y_lower])
    coords = np.column_stack((x, y))

    # Random rotation
    theta = np.random.uniform(0, 2*np.pi)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    coords_rot = coords @ rotation_matrix.T

    # Translate to center
    coords_rot += np.array(center)

    return coords_rot

    return np.column_stack((x, y))
def clean_polygons(polygons, space_size=1000):
    """
    Removes points from polygons that fall outside the [0, space_size] boundary.

    Parameters
    ----------
    polygons : list of np.ndarray
        List of (N, 2) arrays representing polygons.
    space_size : int
        Defines the square 2D boundary (default: 1000).

    Returns
    -------
    cleaned_polygons : list of np.ndarray
        Polygons with only in-bound points retained.
    """
    cleaned_polygons = []

    for poly in polygons:
        # Keep only points within the 0–space_size range
        mask = (
            (poly[:, 0] >= 0) & (poly[:, 0] <= space_size) &
            (poly[:, 1] >= 0) & (poly[:, 1] <= space_size)
        )
        clean_poly = poly[mask]

        # Only keep non-empty polygons
        if len(clean_poly) > 2:
            cleaned_polygons.append(clean_poly)

    return cleaned_polygons

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_rimscore_histogram(rim_score, bins=20):
    """
    Plot a histogram of rim scores.

    Parameters
    ----------
    rim_score : array-like
        List or array of rim scores.
    bins : int
        Number of histogram bins (default: 20).
    """
    rim_score = np.array(rim_score)

    plt.figure(figsize=(6, 4), dpi=120)
    plt.hist(rim_score, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Rim Score')
    plt.ylabel('Count')
    plt.title('Distribution of Rim Scores')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_polygons_threshold(polygons, rim_score, threshold, space_size=1000, show_indices=True):
    """
    Visualize polygons with rim_score thresholding:
    - Green if rim_score < threshold
    - Red if rim_score >= threshold

    Parameters
    ----------
    polygons : list of np.ndarray
        Each polygon is an (N, 2) array of coordinates.
    rim_score : list or array-like
        Numeric values corresponding to each polygon.
    threshold : float
        Threshold to split colors.
    space_size : int
        Size of the 2D plotting space.
    show_indices : bool
        If True, label polygons with rim_score.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Polygon Visualization (Threshold = {threshold})")

    for i, poly in enumerate(polygons):
        x, y = poly[:, 0], poly[:, 1]
        # Close the polygon
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        color = 'green' if rim_score[i] < threshold else 'red'
        ax.plot(x, y, '-', lw=0.8, color='k', alpha=0.5)
        ax.fill(x, y, color=color, alpha=0.6)

        if show_indices:
            cx, cy = np.mean(x[:-1]), np.mean(y[:-1])
            ax.text(cx, cy, f"{rim_score[i]:.2f}", fontsize=6, ha='center', va='center', color='white')

    plt.tight_layout()
    plt.show()

def visualize_polygons(polygons, rim_score, space_size=1000, show_indices=True):
    """
    Visualize a list of polygons (NumPy arrays) in a 2D space,
    with fill color reflecting the rim_score.

    Parameters
    ----------
    polygons : list of np.ndarray
        Each polygon is an (N, 2) array of coordinates.
    rim_score : list or array-like
        Numeric values corresponding to each polygon.
    space_size : int
        Defines the 2D plotting area (default: 1000x1000).
    show_indices : bool
        If True, label polygons by their rim_score value at centroid.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Polygon Visualization (rim_score color-coded)")

    # Normalize rim scores to [0,1] for colormap mapping
    norm = mcolors.Normalize(vmin=min(rim_score), vmax=max(rim_score))
    cmap = cm.viridis

    for i, poly in enumerate(polygons):
        x, y = poly[:, 0], poly[:, 1]
        # Close the polygon for visualization
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        color = cmap(norm(rim_score[i]))
        ax.plot(x, y, '-', lw=0.8, color='k', alpha=0.5)
        ax.fill(x, y, color=color, alpha=0.6)

        if show_indices:
            cx, cy = np.mean(x[:-1]), np.mean(y[:-1])
            ax.text(cx, cy, f"{rim_score[i]:.2f}", fontsize=6, ha='center', va='center', color='white')

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("rim_score", rotation=270, labelpad=10)

    plt.tight_layout()
    plt.show()


# Parameters
n_polygons = 50
space_size = 1000
min_margin = 10  # ensures varying distance from edges
polygons = []

for _ in range(n_polygons):
    # Random center with varying margins
    margin = random.randint(min_margin, 300)
    cx = random.uniform(margin, space_size - margin)
    cy = random.uniform(margin, space_size - margin)

    # Random polygon radius and number of vertices
    radius = random.uniform(30, 120)
    num_vertices = random.randint(5, 10)

    poly = generate_worm_polygon((cx, cy))
    polygons.append(poly)

# Example: print or visualize
for i, p in enumerate(polygons[:3]):
    print(f"Polygon {i}:\n", p[:5], "...")  # show first few points


polygons = clean_polygons(polygons, 1000)



scores = []
for i in polygons:
     scores.append(rim_score(i,[1000,1000]))

visualize_polygons(polygons,scores)
plot_rimscore_histogram(scores)
visualize_polygons_threshold(polygons,scores,0.09)
