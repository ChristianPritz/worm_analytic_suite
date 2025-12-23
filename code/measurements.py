import shutil,copy, joblib,tempfile,cv2,scipy,csv, json,os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import splrep, splev
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import networkx as nx
from networkx.exception import NetworkXNoPath
from skimage.draw import polygon as sk_polygon
from shapely.geometry import Polygon, Point, LineString
from pathlib import Path


from scipy.signal import savgol_filter

from IPython import embed
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter

from skimage.morphology import medial_axis
from skimage.graph import route_through_array
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from classifiers import classify,angle_histogram
import importlib.resources as r


def analyze_annotations(areas_json, points_csv, image_dir,settings, analysis_func):
    """
    Iterate over all area annotations, match with closest head/tail points,
    and run a user-defined analysis function.
    
    Parameters
    ----------
    areas_json : str
        Path to JSON file containing area annotations.
    points_csv : str
        Path to CSV file containing point annotations.
    image_dir : str
        Path to image directory.
    analysis_func : callable
        Function that receives (area, head_coords, tail_coords, image_path)
        and returns a dict, list, or scalar with analysis results.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with area_id, image_name, matched points, coordinates, and analysis results.
    """
    with open(areas_json, 'r') as f:
        data = json.load(f)

    points_df = pd.read_csv(points_csv,header=None)
    results = []

    for img_data in data['images']:
        img_name = Path(img_data['file_name']).name
        img_path = Path(image_dir) / img_name
        img_id = img_data['id']
        

        img_areas = [a for a in data['annotations'] if a['image_id'] == img_id]
        img_points = points_df[points_df.iloc[:,3] == img_name]

        for area in img_areas:
            area_id = area['id']
            coords = np.array(area['segmentation'][0]).reshape(-1, 2)



            head, tail,ol_head,ol_tail = match_head_tail(coords, img_points)
            if head is None or tail is None:
                continue
     
            
            #DEBUG FLAG: ORDER OF INDICES MIGHT BE REVERSED
            # Prepare coordinates for storage and function call
            head_coords = (float(head.iloc[1]), float(head.iloc[2]))
            tail_coords = (float(tail.iloc[1]), float(tail.iloc[2]))

            # Run user analysis function
            analysis_out = analysis_func(area, head_coords, tail_coords,ol_tail, str(img_path))

            # Format output
            if isinstance(analysis_out, dict):
                entry = {**analysis_out}
            elif isinstance(analysis_out, (list, np.ndarray)):
                entry = {f"value_{i}": v for i, v in enumerate(analysis_out)}
            else:
                entry = {"value": analysis_out}


 
            entry.update({
                "area_id": area_id,
                "image_name": img_name,
                "head_id": head.get('id', None),
                "tail_id": tail.get('id', None),
                "head_x": head_coords[0],
                "head_y": head_coords[1],
                "tail_x": tail_coords[0],
                "tail_y": tail_coords[1],
                "label_id":area["category_id"]
            })
            results.append(entry)

    return pd.DataFrame(results)




    
def visualize_worms_threshold(coords, rim_score, threshold, space_size=1000, show_indices=True):
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

    for i, poly in enumerate(coords):
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

def visualize_worms(coords, rim_score, space_size=1000, show_indices=True):
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

    for i, poly in enumerate(coords):
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



#-----------------------------------------------------------------------------#
#
#   A N A L Y T I C   F U N C T I O N S 
#
#-----------------------------------------------------------------------------#




def rim_score(coords,shapes):
    def edgeness(pts,size):
        spread = pts.max() - pts.min()
        distA = pts.min()
        distB = size-pts.max()
        rim_dist = np.min([distA,distB])+1
        
        #print(rim_dist)
        #ptsN = pts/(spread-rim_dist)
        #etric = np.median((np.square((pts-pts.min())/spread))/np.square(rim_dist))
        metric = np.median(((pts-pts.min())/spread)/rim_dist)
        return metric
    
    
    edge_x = edgeness(coords[:,0],shapes[0])
    edge_y = edgeness(coords[:,1],shapes[1])
    
    return np.max([edge_x,edge_y])
    
    
def detect_major_edges(image, granularity=1.0):
    """
    Detects the major edges in an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (RGB or grayscale).
    granularity : float, optional
        Controls the sensitivity or granularity of edge detection.
        Smaller values (e.g. 0.5) detect finer edges.
        Larger values (e.g. 2.0) smooth the image more and keep only major edges.

    Returns
    -------
    edge_map : np.ndarray
        Binary image (uint8) showing major edges (255=edges, 0=background).
    """

    # --- 1. Convert to grayscale if needed ---
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # --- 2. Smooth image based on granularity ---
    # Larger granularity -> stronger Gaussian blur -> fewer edges
    sigma = 1.5 * granularity
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # --- 3. Adaptive thresholds for Canny based on granularity ---
    # Fine control: higher granularity raises thresholds (detects only strong edges)
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.66 * granularity) * v))
    upper = int(min(255, (1.0 + 0.66 * granularity) * v))

    # --- 4. Perform Canny edge detection ---
    edges = cv2.Canny(blurred, lower, upper)

    # --- 5. Optionally clean up small specks with morphological ops ---
    kernel_size = max(1, int(2 * granularity))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges_clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges_clean


def find_head_tail(c, dS, bloc_int, debug=False):
    """
    Identify head and tail of a worm outline based on contour geometry.
    Replaces sklearn with numpy/scipy equivalents.

    Parameters
    ----------
    imgArray : np.ndarray
        Input image (binary or grayscale).
    dS : float
        Simplification factor for polygon reduction.
    bloc_int : int
        Local neighborhood size for curvature estimation.
    debug : bool, optional
        If True, shows diagnostic plots.
    """

    # --- Helper: cyclic index shifting ---
    def indexCycling(n, arr, shift):
        return (arr + shift) % n

    # --- Helper: block of neighbors before and after ---
    def block_before_after(values, displ):
        arr = np.arange(len(values))
        centre = displ
        cBlock = np.empty((1 + 2 * displ, len(arr)))
        cBlock[centre, :] = values
        for i in range(1, displ + 1):
            after_arr = indexCycling(len(arr), arr, i)
            before_arr = indexCycling(len(arr), arr, -i)
            cBlock[centre - i, :] = values[before_arr]
            cBlock[centre + i, :] = values[after_arr]
        return cBlock

    # --- Helper: local pairwise distances (no sklearn) ---
    def local_distances(x, y, h, t, bloc_int):
        def pairwise_dist(A):
            # Equivalent to sklearn.metrics.pairwise_distances
            diff = A[:, None, :] - A[None, :, :]
            return np.sqrt(np.sum(diff**2, axis=-1))

        x_bloc = block_before_after(x, bloc_int)
        y_bloc = block_before_after(y, bloc_int)

        # Head distances
        H = np.stack([x_bloc[:, h], y_bloc[:, h]], axis=1)
        dm = pairwise_dist(H)
        h_sum = np.sum(dm**2)

        # Tail distances
        T = np.stack([x_bloc[:, t], y_bloc[:, t]], axis=1)
        dm = pairwise_dist(T)
        t_sum = np.sum(dm**2)

        return h_sum, t_sum

    # --- Helper: circular angle wrapping ---
    def wrapPi(angles):
        return (angles + np.pi) % (2 * np.pi) - np.pi

    # --- Helper: convert OpenCV contour list to array ---
    def contours2array(contours):
        return np.vstack([cnt.reshape(-1, 2) for cnt in contours])

    # --- Helper: select two peaks with max Euclidean distance ---
    def largest_dist_pks(xps, yps, pk_hgts, topk=4):
        idx = np.flip(np.argsort(pk_hgts))[:topk]
        X, Y = xps[idx], yps[idx]
        D = np.sqrt((X[:, None] - X[None, :])**2 + (Y[:, None] - Y[None, :])**2)
        i, j = np.unravel_index(np.argmax(D), D.shape)
        return idx[i], idx[j]



    # --- Simplify polygon ---
    poly = Polygon(c)
    poly2 = poly.simplify(dS)
    x, y = np.asarray(poly2.exterior.coords.xy)

    # --- Compute geometry ---
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    angles = np.arctan2(np.diff(y), np.diff(x))
    ang = wrapPi(angles)

    # --- Local curvature and distance ---
    cAngs = block_before_after(ang, bloc_int)
    cDist = block_before_after(distances, bloc_int)
    sumDist = np.sum(cDist, axis=0)
    localAng = scipy.stats.circstd(cAngs, axis=0)

    if len(localAng) > 100 or len(localAng) < 40:
        print("Warning: unusual number of local angles =", len(localAng))

    # --- Find curvature peaks (candidate head/tail) ---
    mmers = localAng[-3:]
    inputAng = np.hstack((mmers, localAng))
    pks, _ = scipy.signal.find_peaks(inputAng)
    pks -= 3  # correct offset
    pk_hgts = localAng[pks]
    idx = np.flip(np.argsort(pk_hgts))

    # --- Extract spatial positions ---
    x_ang = x[:-1] + np.diff(x) / 2
    y_ang = y[:-1] + np.diff(y) / 2
    xps, yps = x_ang[pks], y_ang[pks]

    # --- Identify head/tail candidates ---
    h, t = largest_dist_pks(xps, yps, pk_hgts, 4)

    # --- Refine using local distance context ---
    h_sum, t_sum = local_distances(x_ang, y_ang, h, t, bloc_int + 2)

    # --- Assign based on curvature/distance rule ---
    # (tail tends to have higher curvature)
    if pk_hgts[h] > pk_hgts[t]:
        head_idx, tail_idx = t, h
    else:
        head_idx, tail_idx = h, t

    head = [xps[head_idx], yps[head_idx]]
    tail = [xps[tail_idx], yps[tail_idx]]

    # --- Optional debug plots ---
    if debug:
        
        fig, ax = plt.subplots()
        ax.plot(inputAng)
        ax.scatter(pks + 3, pk_hgts)
        plt.title("Local angle curvature peaks")
        plt.show

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.scatter(*head, color="r", label="head")
        ax.scatter(*tail, color="b", label="tail")
        ax.legend()
        ax.axis("equal")
        plt.show()

    return head, tail


def color_deconvolution(image_path, color_matrix, output_dir, prefix="c"):
    """
    Performs color deconvolution on an RGB image using a custom color ratio matrix.
    
    Parameters
    ----------
    image_path : str
        Path to the input RGB image.
    color_matrix : np.ndarray
        3x3 matrix defining the stain color vectors (each column normalized).
    output_dir : str
        Directory where the output channel images will be saved.
    prefix : str, optional
        Filename prefix for output images (default: "channel").
    """
    _,basename=os.path.split(image_path)
    basename=basename[1:-4]
    # --- 1. Load image ---
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img.astype(np.float32) + 1e-6  # avoid log(0)

    # --- 2. Convert RGB to optical density (OD) space ---
    od = -np.log(img_float / 255.0)

    # --- 3. Normalize and invert color matrix ---
    # Ensure color vectors are unit length
    color_matrix = np.array(color_matrix, dtype=np.float32)
    color_matrix /= np.linalg.norm(color_matrix, axis=0, keepdims=True)
    inv_matrix = np.linalg.inv(color_matrix)

    # --- 4. Perform color deconvolution ---
    separated = np.dot(od.reshape((-1, 3)), inv_matrix)
    separated = separated.reshape(img.shape)

    # --- 5. Convert to grayscale images ---
    separated = np.clip(separated, 0, None)
    separated = separated / separated.max(axis=(0,1))  # normalize 0–1

    # --- 6. Save results ---
    os.makedirs(output_dir, exist_ok=True)
    for i in range(3):
        #channel_img = (1 - separated[:, :, i]) * 255  # invert for better contrast
        channel_img = make_8bit_img(separated[:, :, i])
        out_path = os.path.join(output_dir, f"{basename}_{prefix}_{i+1}.png")
        cv2.imwrite(out_path, channel_img.astype(np.uint8))
        print(f"Saved: {out_path}")

    return separated,img


def coco_segmentation_to_array(area):
    """
    Convert COCO-style segmentation to an (N, 2) NumPy array.

    Parameters
    ----------
    area : dict or list
        A COCO-style area annotation. Usually has the form:
        area['segmentation'][0] = [x1, y1, x2, y2, ..., xN, yN]

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with float coordinates [[x1, y1], [x2, y2], ...]
    """
    if isinstance(area, dict) and 'segmentation' in area:
        seg = area['segmentation'][0]
    elif isinstance(area, (list, np.ndarray)):
        seg = area
    else:
        raise ValueError("Input must be a COCO-style area dict or a coordinate list.")

    coords = np.array(seg, dtype=float).reshape(-1, 2)
    return coords


def get_centerline_wrapper(area_coords, head_coords, ol_tail):
    selfoverlap = check_self_overlap(area_coords)
    
    
    try:
        if selfoverlap:
            centerline,_,area = get_worm_centerline_sliding(area_coords, head_coords, ol_tail, window_size=5, multiplier=1, smooth_win=15, smooth_poly=3, plot=True, debug=True)

        else: 
            centerline,_,area = get_worm_centerline(area_coords, plot=False, padding=10)
    except:
        centerline,area = np.nan, np.nan
        
    return centerline, area

def analyze_thickness(area, head_coords, tail_coords,ol_tail, image_path):

    area_coords =coco_segmentation_to_array(area)
    
    

    
    print(area_coords.shape[0])
    
    centerline,area = get_centerline_wrapper(area_coords, head_coords, ol_tail)
    
    #head_coords, tail_coords = find_head_tail(area_coords, 1, int(area_coords.shape[0]/10), debug=True) #settings
    
    
    
    percents=np.arange(0.05,1,0.05)
    error = {}
    for i in percents:
        error[i]=np.nan
    
    if  np.all(np.isnan(centerline)):
        thicks = {}
        for i in percents:
            thicks[i] = np.nan
        length = np.nan
    else:
        try: 
            thicks,length = measure_tube_thickness(area_coords,centerline,head_coords,tail_coords,percents = percents, debug=True)
        except:
            thicks = error
            length = np.nan
    output = {}

    for i in percents:
        
        output['percent_'+str(np.round(i,2))] = thicks[i]
    output["length"] = length
    output["area"] = area
    return output



def find_area_endpoints(polygon_coords):
    """
    Return the two endpoints (extreme coordinates) along the longest axis
    of the area polygon.
    """
    poly = Polygon(polygon_coords)
    min_rect = poly.minimum_rotated_rectangle
    rect_coords = np.array(min_rect.exterior.coords)
    edge_lengths = np.linalg.norm(np.diff(rect_coords, axis=0), axis=1)
    i = np.argmax(edge_lengths)
    return rect_coords[i], rect_coords[i + 1]

def match_head_tail(area_coords, points_df):
    """
    Match predicted area endpoints to user-annotated points (head/tail).
    - area_coords: polygon or contour coordinates (Nx2)
    - points_df: DataFrame with columns [label, x, y]
    
    Returns:
        head_row, tail_row (as pandas Series)
    """
    if len(points_df) == 0:
        return None, None

    # Find predicted head/tail endpoints
    p1, p2 = find_head_tail(area_coords, 1, int(area_coords.shape[0]/10), debug=True)

    def closest_point(target):
        """Return closest annotated point and its label."""
        dists = np.sqrt((points_df.iloc[:, 1] - target[0])**2 + 
                        (points_df.iloc[:, 2] - target[1])**2)
        dists = dists.reset_index(drop=True)
        closest_idx = dists.idxmin()
        return points_df.iloc[closest_idx]

    # Find nearest annotated points to each predicted endpoint
    pt1 = closest_point(p1)
    pt2 = closest_point(p2)

    # Extract user labels
    label1 = pt1.iloc[0].lower().strip()
    label2 = pt2.iloc[0].lower().strip()

    # Ensure only one head and one tail assignment
    if label1 == label2:
        #raise ValueError(f"⚠️ Both matched points are labeled '{label1}'. Check annotations!")
        print(f"⚠️ Both matched points are labeled '{label1}'. Check annotations!")
    # Assign head/tail based on user label
    if label1 == "head" and label2 == "tail":
        head, tail = pt1, pt2
    elif label1 == "tail" and label2 == "head":
        head, tail = pt2, pt1
    else:
        # If one or both labels are missing or nonstandard
        print(f"⚠️ Unexpected or missing user labels: {label1}, {label2}. Assigning by proximity.")
        head, tail = pt1, pt2  # fallback

    return head, tail, p1,p2

def make_8bit_img(mat):
    mat = copy.copy(mat)                # copy
    mat = mat - np.min(mat)             # optional: shift to start at 0
    mat = mat / np.max(mat)             # normalize to 0-1
    mat = np.round(mat * 255)           # scale to 0-255
    img = mat.astype(np.uint8)  
    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap='gray_r')  # grayscale colormap
    cbar = fig.colorbar(im, ax=ax)            # add colorbar
    cbar.set_label('Intensity')               # optional label
    plt.show()    
    return img


def analyze_image_with_annotations(image_path, annotations, color_matrix,output_dir,settings,debug=True):
    """
    Deconvolves image and measures summed intensity per annotation.
    
    Parameters
    ----------
    image_path : str
        Path to the input image.
    annotations : list
        List of annotation dicts (from JSON) for this image.
    color_matrix : np.ndarray
        3x3 color deconvolution matrix.
    
    Returns
    -------
    list of dict
        [{'area_id': ..., 'intensity': ...}, ...]
    """
    _,basename=os.path.split(image_path)
    basename=basename[1:-4]
    separated,orig_img = color_deconvolution(image_path, color_matrix,output_dir)
  
    red_channel = make_8bit_img(separated[:,:,0])
    green_channel = make_8bit_img(separated[:,:,1])
    blue_channel = make_8bit_img(separated[:,:,2])
    
    h, w = red_channel.shape
    results = []

    for anno in annotations:
        area_id = anno['id']
        
        coords = np.array(anno['segmentation'][0]).reshape(-1, 2).astype(np.int32)
   

        try:
            #centerline,area = get_centerline_wrapper(coords, head_coords, ol_tail)
            
            centerline,length,area = get_worm_centerline(coords, plot=debug, padding=10)
    
            centerline = smooth_pts(centerline, win=31, poly=3)
            
            s_im=straighten(red_channel, centerline, int(length*0.15))
            s_im_orig=straighten(orig_img, centerline, int(length*0.15))

            out_path = os.path.join(output_dir, f"red_ch_{basename}_im_{area_id}.png")
            cv2.imwrite(out_path, s_im)
            
            out_path = os.path.join(output_dir, f"orig_im_{basename}_{area_id}.png")
            s_im_orig = s_im_orig[:,:,[2,1,0]]
            cv2.imwrite(out_path, s_im_orig.astype(np.uint8))
            
            
            
            # Create binary mask for this annotation
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [coords], 1)

            # Sum up intensities inside area
            total_intensity = np.sum(red_channel[mask == 1])

            results.append({
                "area_id": area_id,
                "intensity": total_intensity
            })
        except:
            print("[WARNING] Tracing cernterline failed")
    return results


def analyze_staining_from_json(areas_json, image_dir, color_matrix,settings, df=None,output_dir=None):
    """
    Iterates over all images in the annotation JSON, applies color deconvolution-based analysis,
    and collects intensity values per area.

    Parameters
    ----------
    areas_json : str
        Path to JSON file containing area annotations.
    image_dir : str
        Directory containing the images.
    color_matrix : np.ndarray
        3x3 color deconvolution matrix.
    df : pd.DataFrame or None
        Optional existing DataFrame to merge results into.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'area_id', 'image_name', and 'intensity' columns.
    """
    with open(areas_json, 'r') as f:
        data = json.load(f)

    all_results = []

    for img_entry in data['images']:
        img_name = Path(img_entry['file_name']).name
        img_path = Path(image_dir) / img_name
        img_id = img_entry['id']

        annotations = [a for a in data['annotations'] if a['image_id'] == img_id]
        if not annotations:
            continue
        

        # Run analytic function
        image_results = analyze_image_with_annotations(str(img_path), annotations, color_matrix,output_dir,settings)

        # Add image name to each entry
        for r in image_results:
            r['image_name'] = img_name

        all_results.extend(image_results)

    results_df = pd.DataFrame(all_results)



    # Merge or return
    if df is not None:
        merged_df = df.merge(results_df, on="area_id", how="left")
        return merged_df
    else:
        return results_df




# ------------------------- Annotation merging function -----------------------

def collect_annotations(settings,anno_dir='annotations', image_dir='images', color_csv='class_colors.csv', output_dir='output',
                        output_point_csv=None, output_area_json=None):
    """Collects and merges annotation CSVs and JSONs in `anno_dir` into two files
    (points CSV and areas JSON) saved in `output_dir`. Only annotations that refer
    to images present in `image_dir` are kept.

    Parameters
    ----------
    anno_dir: str - directory containing many .csv and .json annotation files
    image_dir: str - directory containing images that should be referenced
    color_csv: str - CSV mapping class names to global IDs and colors
    output_dir: str - directory where merged outputs are written (created if needed)
    output_point_csv: str - optional custom output CSV path
    output_area_json: str - optional custom output JSON path

    Returns
    -------
    dict with keys: point_csv, area_json
    """
    os.makedirs(output_dir, exist_ok=True)
    if output_point_csv is None:
        output_point_csv = os.path.join(output_dir, 'annotations_points.csv')
    if output_area_json is None:
        output_area_json = os.path.join(output_dir, 'annotations_areas.json')

    # 1) Load color CSV and build categories mapping
    color_df = pd.read_csv(color_csv, header=None)
    # Expectation: col0=name, col1-3=RGB, col4=global_id
    name_to_global_id = dict(zip(color_df[0], color_df[4].astype(int)))

    merged_points = []

    # Read CSVs in annotations folder; they may or may not have a header
    for fname in os.listdir(anno_dir):
        fpath = os.path.join(anno_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.lower().endswith('.csv'):
            try:
                df = pd.read_csv(fpath, header=None)
                merged_points.append(df)
            except Exception as e:
                print(f"⚠️ Skipping CSV {fpath}: {e}")

    # Combine points and filter by images present
    if merged_points:
        all_points_df = pd.concat(merged_points, ignore_index=True)
        # If the CSV has at least 4 columns we assume col3 is image_name
        if all_points_df.shape[1] < 4:
            raise ValueError('Point CSVs must contain at least 4 columns (class,x,y,image_name,...)')
        # Normalize columns
        all_points_df = all_points_df.rename(columns={0:'class', 1:'x', 2:'y', 3:'image_name'})
        # Filter rows where image exists in image_dir
        existing_images = set(os.listdir(image_dir))
        filtered_points = all_points_df[all_points_df['image_name'].isin(existing_images)].copy()
        # Save
        filtered_points.to_csv(output_point_csv, header=False, index=False)
        print(f'✅ Points merged and filtered to images present. Saved to: {output_point_csv}')
    else:
        # Empty points file
        pd.DataFrame(columns=['class','x','y','image_name']).to_csv(output_point_csv, header=False, index=False)
        print('ℹ️ No point CSVs found; wrote empty points CSV.')

    # Now process JSONs (COCO-like)
    merged_json = {"images": [], "annotations": [], "categories": []}
    image_id_counter = 1
    ann_id_counter = 1

    # We will build categories from color_csv (ensures global ids)
    categories = []
    for name, gid in name_to_global_id.items():
        categories.append({"id": int(gid), "name": name, "supercategory": ""})
    merged_json['categories'] = copy.deepcopy(categories)

    # Build set of allowed filenames (images present)
    allowed_images = set(os.listdir(image_dir))

    # Iterate JSON files and include only images that are present; remap ids
    for fname in os.listdir(anno_dir):
        fpath = os.path.join(anno_dir, fname)
        if not os.path.isfile(fpath) or not fname.lower().endswith('.json'):
            continue
        try:
            # Clean up class ids first using class_clean_up (will overwrite original JSON)
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            data = class_clean_up(data, color_csv)
        except Exception as e:
            print(f"⚠️ Skipping JSON {fpath}: {e}")
            continue

        # Map old image id -> new image id for images that exist in image_dir
        img_id_map = {}
        for img in data.get('images', []):
            fname_img = img.get('file_name')
            if fname_img in allowed_images:
                new_img = copy.deepcopy(img)
                new_img['id'] = image_id_counter
                # keep file_name and width/height if present
                img_id_map[img['id']] = image_id_counter
                merged_json['images'].append(new_img)
                image_id_counter += 1

        # Add annotations that refer to included images
        for ann in data.get('annotations', []):
            old_img_id = ann.get('image_id')
            if old_img_id in img_id_map:
                new_ann = copy.deepcopy(ann)  
                new_ann['id'] = ann_id_counter
                new_ann['image_id'] = img_id_map[old_img_id]
                # ensure the category_id is a global id (class_clean_up should have already done it)
                merged_json['annotations'].append(new_ann)
                ann_id_counter += 1

    
     
    merged_json_clean,scores = apply_filtering(merged_json,settings)

    print(
    f"🔍 Before filtering: {len(merged_json['annotations'])} annotations\n"
    f"✅ After filtering:  {len(merged_json_clean['annotations'])} annotations"
    )

    # # Save merged JSON
    # for idx,i in enumerate(merged_json_clean["annotations"]):
    #     print(i["id"])
    #     coords = np.array(i["segmentation"])
    #     coords = coords.reshape((int(coords.shape[1]/2),2))
    #     fig,ax = plt.subplots()
    #     ax.plot(coords[:,0],coords[:,1])
    #     plt.title(str())
    #     plt.show()

    save_json_atomically(merged_json_clean, output_area_json)
    return {'point_csv': output_point_csv, 'area_json': output_area_json}


def save_json_atomically(data, output_path):
    # Write to a temporary file first

    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        temp_name = tmp.name

    # Replace the final file atomically
    shutil.move(temp_name, output_path)

def apply_filtering(coco_json, settings):
    """
    Filters annotations based on rim_score.
    Returns a NEW dict — no deepcopies from original.
    """

    # Build clean JSON
    new_json = {
        "images": copy.deepcopy(coco_json["images"]),
        "categories": copy.deepcopy(coco_json["categories"]),
        "annotations": []
    }

    image_sizes = {
        img["id"]: (img["width"], img["height"])
        for img in new_json["images"]
    }

    control_coords = []
    control_scores = []
    clean_scores = []
    #load the badguy classifier... 
    model_path = Path(__file__).resolve().parent / "RF_classifier.pkl"
    model = joblib.load(model_path)
    
    #model = joblib.load("RF_classifier.pkl")

    # iterate over ORIGINAL annotations
    for ann in coco_json["annotations"]:
        img_id = ann["image_id"]

        if "segmentation" not in ann or not isinstance(ann["segmentation"], list):
            continue

        coords = np.array(ann["segmentation"][0]).reshape(-1, 2)
        image_size = image_sizes.get(img_id, None)

        if image_size is None:
            continue


        #filtering the instances for rimness and shape abberations.............
        score = rim_score(coords, image_size)
        hist = angle_histogram(coords)
        badguy,_ = classify(hist.reshape((1,hist.size)),model=model)
        
    
        control_coords.append(coords)
        control_scores.append(score)
        if settings['use_classifier']:
            if score <= settings["rim_score_cutoff"] and badguy[0]==0:
                new_json["annotations"].append(copy.deepcopy(ann))
                clean_scores.append(score)
        else:
            if score <= settings["rim_score_cutoff"]:
                new_json["annotations"].append(copy.deepcopy(ann))
                clean_scores.append(score)

    if settings["debug"] and len(control_coords)>0:

        visualize_worms_threshold(
            control_coords, control_scores, settings["rim_score_cutoff"],
            space_size=image_size[0], show_indices=True
        )
    

    
    return new_json,clean_scores
    


# ---------------------------- class_clean_up (unchanged) ---------------------

def class_clean_up(data, color_csv):
    """
    Returns a cleaned COCO JSON dict where category_id values are replaced
    with the global category IDs defined in `color_csv`.

    Parameters
    ----------
    data : dict
        COCO JSON loaded via json.load()
    color_csv : str
        Path to CSV mapping: name, R, G, B, global_id

    Returns
    -------
    dict
        Cleaned COCO JSON (same structure as input but with updated category_ids)
    """

    # Load the color map CSV
    color_df = pd.read_csv(color_csv, header=None)
    # Expectation: col0 = category_name, col4 = global_category_id
    name_to_global_id = dict(zip(color_df[0], color_df[4].astype(int)))

    # Build mapping: local category id -> name
    local_id_to_name = {}
    for cat in data.get("categories", []):
        local_id_to_name[cat["id"]] = cat["name"]

    cleaned = {
        "images": copy.deepcopy(data.get("images", [])),
        "annotations": [],
        "categories": [],
    }

    # New categories are driven by color CSV (global id list)
    cleaned["categories"] = [
        {"id": int(gid), "name": name, "supercategory": ""}
        for name, gid in name_to_global_id.items()
    ]

    # Fix category_id in annotations
    for ann in data.get("annotations", []):
        old_cat_id = ann["category_id"]

        if old_cat_id not in local_id_to_name:
            continue

        category_name = local_id_to_name[old_cat_id]

        # skip unknown classes (not in CSV)
        if category_name not in name_to_global_id:
            continue

        new_ann = copy.deepcopy(ann)
        new_ann["category_id"] = int(name_to_global_id[category_name])
        cleaned["annotations"].append(new_ann)

    return cleaned



def smooth_pts(pts, win=31, poly=3):
    pts = np.asarray(pts)
    if len(pts) < win:
        win = len(pts) if len(pts) % 2 == 1 else len(pts)-1

    xs = savgol_filter(pts[:, 0], win, poly)
    ys = savgol_filter(pts[:, 1], win, poly)
    return np.column_stack((xs, ys))



def straighten(image, pts, width, debug=False):
    """
    Attribution: 
    This is a python version of Rolf Harkes' straighten.m 
    https://github.com/rharkes/straighten
    
    Straighten image along a spline through pts.
    
    Parameters
    ----------
    image : ndarray
        2D (H,W) or 3D (H,W,C) image (numpy array).
    pts : (M,2) array-like
        Points along backbone in (x, y) order (same as MATLAB pts(:,1), pts(:,2)).
    width : int
        Number of pixels across the straightened output (like MATLAB 'width').
        The function will sample 'width' pixels across the perpendicular,
        starting half-width before the center and stepping by 1 pixel.
    debug : bool
        If True, returns also (xspline, yspline) for plotting/debugging.
    
    Returns
    -------
    IM2 : ndarray
        Straightened image of shape (len(spline), width) for grayscale
        or (len(spline), width, C) for color.
    (optional) (xs, ys) : arrays of spline coordinates if debug==True
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must be shape (N,2) with columns (x,y)")
    if image.ndim not in (2,3):
        raise ValueError("image must be 2D or 3D numpy array")
    if len(pts) < 2:
        raise ValueError("need at least 2 points in pts")
    # Fit parametric spline (t -> x(t), y(t)) by arc-length parameterization
    xspl, yspl = _fit_spline_for_straightening(pts[:,0], pts[:,1])
    n = len(xspl)
    # Output array
    if image.ndim == 2:
        IM2 = np.zeros((n, width), dtype=image.dtype)
    else:
        IM2 = np.zeros((n, width, image.shape[2]), dtype=image.dtype)

    # For each spline point, compute perpendicular step and sample width pixels
    for ct in range(n):
        if ct == 0:
            dx = xspl[1] - xspl[0]     # delta x (cols)
            dy = yspl[0] - yspl[1]     # note sign matches MATLAB code
        else:
            dx = xspl[ct] - xspl[ct-1]
            dy = yspl[ct-1] - yspl[ct]

        L = np.hypot(dx, dy)
        if L == 0:
            # degenerate, skip or use previous direction
            dx, dy = 1.0, 0.0
            L = 1.0
        dx /= L
        dy /= L

        # start half width away: xStart = x(ct) - (dy*width)/2
        xstart = xspl[ct] - (dy*width)/2.0
        ystart = yspl[ct] - (dx*width)/2.0

        # step along perpendicular: add (dy, dx) each iteration (as in MATLAB)
        if image.ndim == 2:
            for ct2 in range(width):
                val = _get_interpolated_value(image, ystart, xstart)
                IM2[ct, ct2] = val
                xstart += dy
                ystart += dx
        else:
            for ct2 in range(width):
                vals = [_get_interpolated_value(image[..., c], ystart, xstart)
                        for c in range(image.shape[2])]
                IM2[ct, ct2, :] = vals
                xstart += dy
                ystart += dx

    if debug:
        return IM2, (xspl, yspl)
    return IM2


# ---------------- helper routines ----------------



def _get_interpolated_value(IM_channel, y, x):
    """
    Bilinear interpolation of single-channel image IM_channel at floating
    coordinates (y: row, x: col). Outside-of-bounds returns 0. (zero padding)
    """
    h, w = IM_channel.shape
    # floor/ceil
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    # fractional parts
    fx = x - x0
    fy = y - y0

    def samp(ii, jj):
        if ii < 0 or ii >= h or jj < 0 or jj >= w:
            return 0.0
        return float(IM_channel[ii, jj])

    v00 = samp(y0, x0)
    v01 = samp(y0, x1)
    v10 = samp(y1, x0)
    v11 = samp(y1, x1)

    # bilinear
    val = (v00 * (1 - fx) * (1 - fy) +
           v01 * fx * (1 - fy) +
           v10 * (1 - fx) * fy +
           v11 * fx * fy)
    return val


def _fit_spline_for_straightening(x_vals, y_vals):
    """
    Return xspl, yspl: points along a parametric spline resampled so that
    distance between consecutive output points is approximately 1 pixel.

    Approach:
    - parameterize input points by cumulative arc length t (0..total_length)
    - fit cubic splines x(t), y(t) via splrep
    - sample dense points along t with step ~0.5 pixels
    - walk through dense samples and output points such that spacing between
      successive output points is 1 pixel (like the MATLAB code)
    """
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    # compute cumulative arc-length
    diffs = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)
    cumlen = np.concatenate(([0.0], np.cumsum(diffs)))
    total = cumlen[-1]
    if total == 0:
        # all pts identical: return single-point spline
        return np.array([x_vals[0]]), np.array([y_vals[0]])

    # parameter t in [0, total]
    t = cumlen
    # fit splines x(t), y(t) (parametric)
    # require at least 3 points for cubic; fallback to lower-order if needed
    k = 3 if len(t) >= 4 else (len(t) - 1)
    if k < 1:
        # line of two points
        return np.array(x_vals), np.array(y_vals)

    tx = splrep(t, x_vals, k=k)
    ty = splrep(t, y_vals, k=k)

    # sample dense points with approx 0.5 pixel spacing along arc-length
    step = 0.5
    num_samples = max(int(np.ceil(total / step)) + 1, 2)
    t_dense = np.linspace(0.0, total, num_samples)
    x_dense = splev(t_dense, tx)
    y_dense = splev(t_dense, ty)

    # now resample to have distance ~1 between points
    xspl = [x_vals[0]]
    yspl = [y_vals[0]]
    L_acc = 0.0
    ptw = 1  # number of points written (MATLAB variable)
    # iterate dense samples and add points when we've accumulated >=1 pixel
    prev_x = x_dense[0]
    prev_y = y_dense[0]
    for idx in range(1, len(x_dense)):
        cur_x = x_dense[idx]
        cur_y = y_dense[idx]
        dx = cur_x - prev_x
        dy = cur_y - prev_y
        d = np.hypot(dx, dy)
        L_acc += d
        if d == 0:
            prev_x = cur_x
            prev_y = cur_y
            continue
        # if we've gone over the next integer length, add a point on the segment
        while L_acc >= 1.0:
            # fraction back along last segment to put point exactly 1 pixel from last written
            frac = (L_acc - 1.0) / d  # fraction of current small step we must backtrack
            # place new point at cur - frac*(dx,dy)
            new_x = cur_x - frac * dx
            new_y = cur_y - frac * dy
            xspl.append(new_x)
            yspl.append(new_y)
            # we've placed one pixel step; decrease accumulator by 1 and continue
            L_acc -= 1.0
            # after placing, the "previous" virtual point becomes new point; but we stay on same segment
        prev_x = cur_x
        prev_y = cur_y

    return np.array(xspl), np.array(yspl)


def measure_tube_thickness(outline_coords, centerline_coords, start_coord, end_coord,
                           percents=[0.1, 0.25, 0.5, 0.75, 0.9],
                           smooth_window=7, smooth_poly=3,
                           tangent_window=10,
                           debug=False):
    """
    Measure tube thickness along a (smoothed) centerline between start and end points.
    """

    # Smooth centerline
    cx, cy = np.array(centerline_coords).T
    if len(cx) >= smooth_window:
        cx_smooth = savgol_filter(cx, smooth_window, smooth_poly)
        cy_smooth = savgol_filter(cy, smooth_window, smooth_poly)
        centerline_coords = np.column_stack((cx_smooth, cy_smooth))

    # Convert to shapely LineString
    centerline = LineString(centerline_coords)
    outline = Polygon(outline_coords)

    # Find closest points on centerline to start and end
    start_pt = Point(start_coord)
    end_pt = Point(end_coord)
    start_dist = centerline.project(start_pt)
    end_dist = centerline.project(end_pt)

    # Ensure correct order (start before end)
    if end_dist < start_dist:
        start_dist, end_dist = end_dist, start_dist

    # Cut centerline segment
    centerline_segment = centerline.interpolate(start_dist), centerline.interpolate(end_dist)
    cut_line = LineString([
        (centerline.interpolate(d).x, centerline.interpolate(d).y)
        for d in np.linspace(start_dist, end_dist, 200)
    ])
    centerline_coords = np.array(cut_line.coords)

    segment_length = cut_line.length
    total_length = segment_length

    # Ensure direction starts near start_coord
    d0 = start_pt.distance(Point(centerline_coords[0]))
    d1 = start_pt.distance(Point(centerline_coords[-1]))
    if d1 < d0:
        centerline_coords = centerline_coords[::-1]
        cut_line = LineString(centerline_coords)

    thicknesses = {}
    debug_data = []

    # Compute thickness at defined percentages
    for p in percents:
        target_dist = p * total_length
        point = cut_line.interpolate(target_dist)
        center = np.array([point.x, point.y])

        # Local tangent
        dists = np.linspace(
            max(0, target_dist - tangent_window),
            min(total_length, target_dist + tangent_window),
            num=5
        )
        local_pts = np.array([[cut_line.interpolate(d).x,
                               cut_line.interpolate(d).y] for d in dists])
        dx, dy = np.gradient(local_pts[:, 0]), np.gradient(local_pts[:, 1])
        tangent = np.array([np.mean(dx), np.mean(dy)])
        tangent /= np.linalg.norm(tangent)

        # Perpendicular direction
        perp = np.array([-tangent[1], tangent[0]])
        L = total_length * 2
        line = LineString([
            (point.x - perp[0] * L, point.y - perp[1] * L),
            (point.x + perp[0] * L, point.y + perp[1] * L)
        ])

        # Intersect with outline
        inter = outline.boundary.intersection(line)
        if inter.is_empty:
            thicknesses[p] = None
            continue

        pts = []
        if inter.geom_type == "Point":
            pts = [np.array([inter.x, inter.y])]
        elif inter.geom_type == "MultiPoint":
            pts = [np.array([pt.x, pt.y]) for pt in inter.geoms]
        else:
            for geom in inter.geoms:
                if geom.geom_type == "Point":
                    pts.append(np.array([geom.x, geom.y]))
                elif geom.geom_type == "LineString":
                    pts.extend(np.array(geom.coords))

        if len(pts) < 2:
            thicknesses[p] = None
            continue

        # Split into two sides of the perpendicular
        dots = [np.dot(pt - center, perp) for pt in pts]
        left_pts = [pt for pt, d in zip(pts, dots) if d < 0]
        right_pts = [pt for pt, d in zip(pts, dots) if d > 0]
        if not left_pts or not right_pts:
            thicknesses[p] = None
            continue

        left_closest = min(left_pts, key=lambda x: np.linalg.norm(x - center))
        right_closest = min(right_pts, key=lambda x: np.linalg.norm(x - center))
        dist = np.linalg.norm(left_closest - right_closest)

        thicknesses[p] = dist
        debug_data.append((point, left_closest, right_closest))

    # Debug visualization
    if debug:
        fig, ax = plt.subplots(dpi=400)
        ox, oy = zip(*outline.exterior.coords)
        ax.plot(ox, oy, 'k-', lw=0.7, label='Outline')

        cx, cy = zip(*centerline_coords)
        ax.plot(cx, cy, 'r-', lw=0.8, label='Cut centerline')

        # Start and end points
        ax.plot(start_coord[0], start_coord[1], 'yo', ms=8, label='Start point', markeredgecolor='k')
        ax.plot(end_coord[0], end_coord[1], 'co', ms=8, label='End point', markeredgecolor='k')

        # Measurement lines
        for center_point, left_pt, right_pt in debug_data:
            ax.plot([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], 'b-', lw=0.6)
            ax.plot(center_point.x, center_point.y, 'go', ms=4)
            ax.plot(left_pt[0], left_pt[1], 'mo', ms=3)
            ax.plot(right_pt[0], right_pt[1], 'mo', ms=3)

        ax.legend(loc='upper right')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title("Tube thickness analysis (truncated by start/end)")
        plt.show()

    return thicknesses, segment_length



def resample_to_regular_spacing(coords, min_points=220):
    """
    Resample 2D coordinates so that all consecutive points have exactly
    the same distance. Ensures that at least `min_points` are returned.

    Parameters
    ----------
    coords : Nx2 array
        Original polyline coordinates.
    min_points : int (default = 220)
        Minimum number of resampled points.

    Returns
    -------
    resampled_coords : Mx2 array
        Resampled coordinates with uniform spacing along the curve.
        M >= min_points
    """
    coords = np.asarray(coords)
    if coords.shape[0] < 2:
        return coords  # nothing to do

    diffs = np.diff(coords, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cumdist = np.hstack([0, np.cumsum(dists)])

    total_length = cumdist[-1]

    # original spacing based on smallest segment
    spacing = np.min(dists)
    n_segments = int(np.round(total_length / spacing))

    # enforce minimum number of points
    if n_segments + 1 < min_points:
        n_segments = min_points - 1  # so we have min_points total coords

    # recompute the exact spacing
    spacing_exact = total_length / n_segments

    # new cumulative distances (uniform)
    new_cumdist = np.linspace(0, total_length, n_segments + 1)

    # interpolate x and y separately
    fx = interp1d(cumdist, coords[:, 0], kind='linear')
    fy = interp1d(cumdist, coords[:, 1], kind='linear')

    resampled_coords = np.column_stack((fx(new_cumdist), fy(new_cumdist)))
    return resampled_coords

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

def get_worm_centerline_sliding(poly, head, tail, window_size=5, smooth_win=15, smooth_poly=3, multiplier=3, plot=False, debug=False, padding=5):
    
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

    import numpy as np
    from scipy.spatial.distance import cdist
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    from skimage.draw import polygon as sk_polygon

    poly = np.asarray(poly)
    head = np.asarray(head)
    tail = np.asarray(tail)

    poly = resample_to_regular_spacing(poly)  # helper function you already have

    # --- find nearest poly indices to head and tail ---
    dist_head = np.linalg.norm(poly - head, axis=1)
    dist_tail = np.linalg.norm(poly - tail, axis=1)
    idx_head = np.argmin(dist_head)
    idx_tail = np.argmin(dist_tail)

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

    # --- compute centerline length ---
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
    if plot or debug:
        plt.figure(figsize=(8,6))
        plt.plot(poly[:,0], poly[:,1], 'k-', label='Outline')
        if debug:
            plt.plot(half1[:,0], half1[:,1], 'r.-', label='Half 1 (template)')
            plt.plot(half2[:,0], half2[:,1], 'b.-', label='Half 2')
        plt.plot(midline[:,0], midline[:,1], 'k-', lw=2, label='Centerline')
        plt.scatter(head[0], head[1], c='g', s=50, label='Head')
        plt.scatter(tail[0], tail[1], c='m', s=50, label='Tail')
        plt.axis('equal')
        plt.legend()
        plt.show()

    return midline, length, area

def check_self_overlap(poly):
    """
    Check if a polygon/polyline self-overlaps.

    Parameters
    ----------
    poly : Nx2 array
        Coordinates of the polygon/polyline (closed or open).

    Returns
    -------
    overlap : bool
        True if the shape self-overlaps, False otherwise.
    """
    poly = np.asarray(poly)
    n = len(poly)

    def segments_intersect(p1, p2, q1, q2):
        """Check if line segments (p1,p2) and (q1,q2) intersect."""
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return ccw(p1,q1,q2) != ccw(p2,q1,q2) and ccw(p1,p2,q1) != ccw(p1,p2,q2)

    # Check all non-adjacent segment pairs
    for i in range(n-1):
        for j in range(i+2, n-1):
            # Skip if the segments share a point
            if i == 0 and j == n-2:  # skip first-last if closed polygon
                continue
            if segments_intersect(poly[i], poly[i+1], poly[j], poly[j+1]):
                return True
    return False

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


def get_worm_centerline_legacy(poly, plot=False, padding=10):
    """
    Extract centerline from an open or closed worm polygon.
    
    Parameters:
    - poly: Nx2 array of xy points (polygon of worm outline)
    - plot: bool, if True plot polygon and centerline
    - padding: int, pixels to pad around polygon in mask
    
    Returns:
    - centerline: Mx2 array of xy coords along centerline (same space as input)
    - length: float, length of centerline in pixels
    - area: int, number of pixels in mask
    """
    # --- ensure np array ---
    poly = np.asarray(poly)
    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])  # close polygon

    # --- shift polygon to positive coords for rasterization ---
    min_xy = poly.min(axis=0)
    poly_shifted = poly - min_xy + padding  # shift + pad margin

    # --- rasterize polygon ---
    max_xy = np.ceil(poly_shifted.max(axis=0)) + padding
    img_shape = (int(max_xy[1])+1, int(max_xy[0])+1)  # rows, cols
    rr, cc = sk_polygon(poly_shifted[:,1], poly_shifted[:,0], img_shape)
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask[rr, cc] = 1
    area = np.sum(mask)

    # --- morphological close ---
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- skeletonize ---
    skeleton = skeletonize(mask > 0)

    # --- build pixel graph ---
    ys, xs = np.where(skeleton)
    G = nx.Graph()
    for x, y in zip(xs, ys):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < mask.shape[1] and 0 <= ny_ < mask.shape[0]:
                    if skeleton[ny_, nx_]:
                        G.add_edge((x, y), (nx_, ny_))

    # --- find endpoints ---
    endpoints = [n for n, d in G.degree() if d == 1]
    if len(endpoints) < 2:
        raise ValueError("Could not find two endpoints in skeleton.")

    # choose farthest apart endpoints
    if len(endpoints) > 2:
        dists = [
            (np.linalg.norm(np.array(a) - np.array(b)), a, b)
            for i, a in enumerate(endpoints)
            for b in endpoints[i+1:]
        ]
        _, end1, end2 = max(dists, key=lambda x: x[0])
    else:
        end1, end2 = endpoints

    # --- find path ---
    try:
        path = nx.shortest_path(G, source=end1, target=end2)
    except NetworkXNoPath:
        raise ValueError("No valid path found in skeleton.")

    # --- convert to xy coords ---
    centerline_shifted = np.array(path, dtype=float)
    centerline_shifted = centerline_shifted[:, [0, 1]]  # (x, y)

    # --- shift back to original coordinate system ---
    centerline = centerline_shifted - padding + min_xy

    # --- compute centerline length ---
    diffs = np.diff(centerline, axis=0)
    length = np.sum(np.linalg.norm(diffs, axis=1))

    # --- plot ---
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(poly[:,0], poly[:,1], 'b-', label='Outline')
        plt.plot(centerline[:,0], centerline[:,1], 'r-', lw=2, label='Centerline')
        plt.axis('equal')
        plt.legend()
        plt.title(f'Centerline length: {length:.2f}px')
        plt.show()

    return centerline, length, area

###############################################################################
#
# Data and string handling
#
###############################################################################
def create_group_labels(df,grps=None,arr=None):
    if grps is None:
        grps = {0:"NG",1:"LH",2:"0G"}
    if arr is None:
        arr = np.array([[1,0,0],[0,1,0],[0,0,1]])
    df["group_identifier"] = "test"
    for i in range(len(df)):
        query = df.iloc[i][['is_NG', 'is_LH', 'is_0G']].to_numpy().astype(np.int64)
        idx = np.where(np.all(arr == query, axis=1))[0]
        df.loc[df.index[i], "group_identifier"] = grps[idx[0]]
    return df

def check_cond(istr):
    marker = [0,0,0,0,0]
    if '_NG' in istr:
        marker[0] = 1
    if '_LH'  in istr:
        marker[1] = 1
    if '_0G'  in istr:
        marker[2] = 1
    
    
    if '_G' in istr:
        marker[3] = int(istr[6])
        
    if '_T' in istr:
        marker[4] = int(istr[9])
        
    out ={'is_NG':marker[0],'is_LH':marker[1],'is_0G':marker[2],
          'generation':marker[3],'trial':marker[4]}
    
    return out

def assign_conditions(df,col_name = "image_name"):
    """
    Applies `check_cond` to every value in df[col_name],
    returns a dict: {original_value: result},
    and extends the dataframe with a new column containing results.
    """
    # compute outputs for each row and build dict
    
    for idx, value in enumerate(df[col_name]):
        m_dict = check_cond(value[0:10])
        print(value[0:10], m_dict)
    
        for key, val in m_dict.items():
            df.loc[idx, key] = val

    return df


def organize_dataset(base_dir):
    """
    Scans recursively for images and annotation files, and moves them into
    subfolders /images and /labels inside the base directory.

    Parameters
    ----------
    base_dir : str
        Path to the directory to scan.
    """

    base = Path(base_dir)
    images_dir = base / "images"
    labels_dir = base / "labels"

    # Create target directories if not exist
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # file extensions
    image_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    label_exts = {".csv", ".json"}

    # --- FIRST PASS: move image files ---
    for root, _, files in os.walk(base):
        for file in files:
            ext = Path(file).suffix.lower()

            if ext in image_exts:
                src = Path(root) / file
                dst = images_dir / file

                # If a file already exists in /images, rename
                if dst.exists():
                    new_name = dst.stem + dst.suffix
                    dst = images_dir / new_name

                print(f"Moving image: {src} -> {dst}")
                shutil.move(str(src), str(dst))

    # --- SECOND PASS: move JSON / CSV (labels) ---
    for root, _, files in os.walk(base):
        for file in files:
            ext = Path(file).suffix.lower()

            if ext in label_exts:
                src = Path(root) / file
                dst = labels_dir / file

                if dst.exists():
                    new_name = dst.stem + dst.suffix
                    dst = labels_dir / new_name

                print(f"Moving label: {src} -> {dst}")
                shutil.move(str(src), str(dst))

    print("\n Done! All images moved to /images and labels moved to /labels.\n")

