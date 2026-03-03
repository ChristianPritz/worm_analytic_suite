#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 14:28:19 2026

@author: wormulon
"""



import os,json, copy, random,shutil,yaml
from pathlib import Path
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_and_interpolate_coco(coco_path, params):
    """
    Smooth and interpolate all polygon segmentations in a COCO file.

    Parameters
    ----------
    coco_path : str
        Path to COCO JSON file.

    params : dict
        {
            "smoothing": {
                "enabled": True,
                "sigma": 2.0
            },
            "interpolation": {
                "mode": "fixed_points",   # "fixed_points" or "fixed_density"
                "num_points": 200,        # used if mode == fixed_points
                "density": 0.1            # points per pixel (e.g. 0.1 → 1 point per 10 px)
            },
            "output_path": None  # optional new file path
        }
    """

    with open(coco_path, "r") as f:
        coco = json.load(f)

    smoothing_enabled = params.get("smoothing", {}).get("enabled", False)
    sigma = params.get("smoothing", {}).get("sigma", 2.0)

    interp_mode = params.get("interpolation", {}).get("mode", "fixed_points")
    num_points = params.get("interpolation", {}).get("num_points", 200)
    density = params.get("interpolation", {}).get("density", 0.1)

    for ann in coco["annotations"]:
        if "segmentation" not in ann:
            continue

        # Skip RLE masks
        if not isinstance(ann["segmentation"], list):
            continue

        polygons = ann["segmentation"]

        new_polygons = []

        for poly in polygons:
            coords = np.array(poly).reshape(-1, 2)

            # Ensure closed polygon
            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])

            x = coords[:, 0]
            y = coords[:, 1]

            # --------------------------------
            # SMOOTHING
            # --------------------------------
            if smoothing_enabled:
                x = gaussian_filter1d(x, sigma=sigma, mode='wrap')
                y = gaussian_filter1d(y, sigma=sigma, mode='wrap')

            smoothed = np.column_stack([x, y])

            # --------------------------------
            # ARC LENGTH PARAMETRIZATION
            # --------------------------------
            deltas = np.diff(smoothed, axis=0)
            seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
            cumulative = np.insert(np.cumsum(seg_lengths), 0, 0)
            total_length = cumulative[-1]

            if total_length == 0:
                continue

            # --------------------------------
            # INTERPOLATION MODE
            # --------------------------------
            if interp_mode == "fixed_points":
                target_n = num_points

            elif interp_mode == "fixed_density":
                target_n = max(int(total_length * density), 10)

            else:
                raise ValueError("Interpolation mode must be 'fixed_points' or 'fixed_density'")

            new_distances = np.linspace(0, total_length, target_n)

            new_x = np.interp(new_distances, cumulative, x)
            new_y = np.interp(new_distances, cumulative, y)

            new_coords = np.column_stack([new_x, new_y]).flatten().tolist()
            new_polygons.append(new_coords)

        ann["segmentation"] = new_polygons

    # --------------------------------
    # SAVE
    # --------------------------------
    output_path = params.get("output_path")
    if output_path is None:
        output_path = coco_path

    with open(output_path, "w") as f:
        json.dump(coco, f)

    print(f"Updated COCO file saved to: {output_path}")




def inspect_and_verify_coco(
    coco_json_path,
    delete_invalid=False,
    shrink_keypoints=False,
):
    """
    Inspects and verifies a COCO annotation file.

    Parameters
    ----------
    coco_json_path : str
        Path to COCO json file.

    delete_invalid : bool
        If True, removes annotations missing segmentation or keypoints.
        If False, only flags them.

    shrink_keypoints : bool
        If True, keeps only first and last keypoint (x,y,v triplets).

    Returns
    -------
    None
        Saves a verified JSON file with suffix '_verified.json'.
    """

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    annotations = coco.get("annotations", [])
    verified_annotations = []
    seen = set()
    removed_count = 0
    flagged_count = 0
    duplicate_count = 0

    for ann in annotations:
        ann_copy = copy.deepcopy(ann)

        # ---------- Check required fields ----------
        has_seg = "segmentation" in ann_copy and ann_copy["segmentation"]
        has_kp = "keypoints" in ann_copy and ann_copy["keypoints"]

        if not (has_seg and has_kp):
            flagged_count += 1
            print(f"[FLAGGED] Annotation ID {ann_copy.get('id')} missing segmentation or keypoints")

            if delete_invalid:
                removed_count += 1
                continue

        # ---------- Shrink keypoints ----------
        if shrink_keypoints and has_kp:
            kps = ann_copy["keypoints"]
            if len(kps) >= 6:  # at least 2 keypoints (x,y,v)
                first = kps[:3]
                last = kps[-3:]
                ann_copy["keypoints"] = first + last
                ann_copy["num_keypoints"] = 2

        # ---------- Detect duplicates ----------
        key = (
            ann_copy.get("image_id"),
            ann_copy.get("category_id"),
            tuple(ann_copy.get("bbox", [])),
            str(ann_copy.get("segmentation")),
            tuple(ann_copy.get("keypoints", [])),
        )

        if key in seen:
            duplicate_count += 1
            print(f"[DUPLICATE] Annotation ID {ann_copy.get('id')}")
            continue

        seen.add(key)
        verified_annotations.append(ann_copy)

    # Replace annotations
    coco["annotations"] = verified_annotations

    # ---------- Save new file ----------
    base, ext = os.path.splitext(coco_json_path)
    new_path = f"{base}_verified{ext}"

    with open(new_path, "w") as f:
        json.dump(coco, f, indent=4)

    print("\n===== COCO INSPECTION COMPLETE =====")
    print(f"Original annotations: {len(annotations)}")
    print(f"Verified annotations: {len(verified_annotations)}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Flagged missing entries: {flagged_count}")
    if delete_invalid:
        print(f"Deleted invalid entries: {removed_count}")
    print(f"Saved verified file to: {new_path}")

def split_coco_annotations(
    input_json,
    output_train_json,
    output_val_json,
    train_ratio=0.8,
    seed=42
):
    random.seed(seed)

    # Load COCO file
    with open(input_json, 'r') as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    # Shuffle images
    random.shuffle(images)

    # Split images
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Get image IDs
    train_image_ids = set(img['id'] for img in train_images)
    val_image_ids = set(img['id'] for img in val_images)

    # Split annotations by image_id
    train_annotations = [
        ann for ann in annotations if ann['image_id'] in train_image_ids
    ]
    val_annotations = [
        ann for ann in annotations if ann['image_id'] in val_image_ids
    ]

    # Create output dicts
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }

    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories
    }

    # Save files
    with open(output_train_json, 'w') as f:
        json.dump(train_coco, f, indent=4)

    with open(output_val_json, 'w') as f:
        json.dump(val_coco, f, indent=4)

    print(f"Train images: {len(train_images)}, annotations: {len(train_annotations)}")
    print(f"Val images: {len(val_images)}, annotations: {len(val_annotations)}")




def convert_coco_json(
    src_img_dir,
    json_path,
    images_dir,
    output_dir,
    base_dir,
    use_segments=True,
    use_keypoints=False,
    copy_images=True,
    label = ''
):
    """
    Convert COCO JSON to YOLO format and optionally copy images, 
    then generate YOLO dataset YAML file in base_dir.

    Parameters
    ----------
    src_img_dir : str
        Directory of original images
    json_path : str
        Path to COCO json file
    images_dir : str
        Directory where YOLO images will be stored
    output_dir : str
        Directory where YOLO labels will be stored
    base_dir : str
        Base folder where the dataset.yaml will be saved
    use_segments : bool
        Use segmentation polygons
    use_keypoints : bool
        Use keypoints
    copy_images : bool
        Copy images into YOLO image folder
    """

    with open(json_path) as f:
        data = json.load(f)

    src_img_dir = Path(src_img_dir)
    output_dir = Path(output_dir)
    images_dir = Path(images_dir)
    base_dir = Path(base_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Create corresponding images directory
    if copy_images:
        images_output_dir = Path(images_dir)
        os.makedirs(images_output_dir, exist_ok=True)

    # Map image_id → image info
    images = {img["id"]: img for img in data["images"]}

    # Map image_id → annotations
    anns = defaultdict(list)
    for ann in data["annotations"]:
        anns[ann["image_id"]].append(ann)

    # Category mapping
    categories = {cat["id"]: i for i, cat in enumerate(data["categories"])}
    class_names = [cat["name"] for cat in data["categories"]]

    print("Category mapping:")
    for cat in data["categories"]:
        print(f"{cat['name']} → {categories[cat['id']]}")

    for img_id, img_info in images.items():

        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        label_path = output_dir / (Path(file_name).stem + ".txt")
        lines = []

        for ann in anns[img_id]:

            cls = categories[ann["category_id"]]
            x, y, w, h = ann["bbox"]

            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w /= width
            h /= height

            if use_keypoints and "keypoints" in ann:

                keypoints = ann["keypoints"]
                kpts_out = []

                for i in range(0, len(keypoints), 3):
                    kx = keypoints[i] / width
                    ky = keypoints[i + 1] / height
                    kv = keypoints[i + 2]
                    kpts_out.extend([kx, ky, kv])

                line = [cls, x_center, y_center, w, h] + kpts_out

            elif use_segments and "segmentation" in ann:

                segments = ann["segmentation"]

                for seg in segments:
                    seg_norm = []
                    for i in range(0, len(seg), 2):
                        px = seg[i] / width
                        py = seg[i + 1] / height
                        seg_norm.extend([px, py])

                    line = [cls] + seg_norm
                    lines.append(" ".join(map(str, line)))
                continue

            else:
                line = [cls, x_center, y_center, w, h]

            lines.append(" ".join(map(str, line)))

        with open(label_path, "w") as f:
            if lines:
                f.write("\n".join(lines))

        # ---------- Copy image ----------
        if copy_images:

            src_img = src_img_dir / file_name
            dst_img = images_output_dir / file_name

            if src_img.exists():
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)
            else:
                print(f"[WARNING] Missing image: {src_img}")

    # ---- Generate dataset.yaml ----
    yamlname = label + "dataset.yaml"
    yaml_path = base_dir / yamlname
    yaml_data = {
        "train": str(Path(images_dir) / ""),
        "val": str(Path(images_dir) / ""),
        "nc": len(class_names),
        "names": class_names
    }

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)

    print(f"\nConversion complete → {output_dir}")
    print(f"YOLO dataset YAML saved → {yaml_path}")
    



def yolo_to_coco_adapter(image_name, yolo_result, json_path):
    """
    Convert Ultralytics YOLO result object to COCO format and append to JSON.

    Parameters
    ----------
    image_name : str
    yolo_result : ultralytics.engine.results.Results
    json_path : str
    """

    json_path = Path(json_path)

    # ------------------------------------------------------------
    # 1️⃣ Load or create COCO structure
    # ------------------------------------------------------------
    if not json_path.exists():
        print(f"[INFO] COCO file not found. Creating new file at {json_path}")

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)

    else:
        with open(json_path, "r") as f:
            coco_data = json.load(f)

    # ------------------------------------------------------------
    # 2️⃣ Image ID handling
    # ------------------------------------------------------------
    existing_images = {img["file_name"]: img for img in coco_data["images"]}

    height, width = yolo_result.orig_shape

    if image_name in existing_images:
        image_id = existing_images[image_name]["id"]
    else:
        existing_ids = [img["id"] for img in coco_data["images"]]
        image_id = max(existing_ids) + 1 if existing_ids else 1

        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": int(width),
            "height": int(height)
        })

    # ------------------------------------------------------------
    # 3️⃣ Sync categories
    # ------------------------------------------------------------
    model_names = yolo_result.names
    existing_cat_ids = {cat["id"] for cat in coco_data["categories"]}

    for class_id, class_name in model_names.items():
        if class_id not in existing_cat_ids:
            coco_data["categories"].append({
                "id": int(class_id),
                "name": class_name,
                "supercategory": "none"
            })

    # ------------------------------------------------------------
    # 4️⃣ Annotation ID handling
    # ------------------------------------------------------------
    existing_ann_ids = [ann["id"] for ann in coco_data["annotations"]]
    next_ann_id = max(existing_ann_ids) + 1 if existing_ann_ids else 1

    # ------------------------------------------------------------
    # 5️⃣ Add detections
    # ------------------------------------------------------------
    boxes = yolo_result.boxes
    masks = yolo_result.masks
    keypoints = yolo_result.keypoints

    if boxes is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for i in range(len(xyxy)):

            x_min, y_min, x_max, y_max = xyxy[i]

            # Detect normalization (0-1)
            if max(xyxy[i]) <= 1.0:
                x_min *= width
                x_max *= width
                y_min *= height
                y_max *= height

            box_w = x_max - x_min
            box_h = y_max - y_min

            annotation = {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": int(classes[i]),
                "bbox": [
                    float(x_min),
                    float(y_min),
                    float(box_w),
                    float(box_h)
                ],
                "area": float(box_w * box_h),
                "iscrowd": 0,
                "score": float(confs[i])
            }

            # ----------------------------------------------------
            # Masks (segmentation)
            # ----------------------------------------------------
            if masks is not None and masks.xy is not None:
                polygon = masks.xy[i]

                # Check normalization
                if np.max(polygon) <= 1.0:
                    polygon[:, 0] *= width
                    polygon[:, 1] *= height

                annotation["segmentation"] = [
                    polygon.flatten().tolist()
                ]

            # ----------------------------------------------------
            # Keypoints
            # ----------------------------------------------------
            if keypoints is not None:
                kpts = keypoints.xy[i].cpu().numpy()

                if np.max(kpts) <= 1.0:
                    kpts[:, 0] *= width
                    kpts[:, 1] *= height

                # COCO expects [x,y,v] triplets
                # YOLO gives only x,y → set visibility = 2
                kpt_list = []
                for x, y in kpts:
                    kpt_list.extend([float(x), float(y), 2])

                annotation["keypoints"] = kpt_list
                annotation["num_keypoints"] = len(kpts)

            coco_data["annotations"].append(annotation)
            next_ann_id += 1

    # ------------------------------------------------------------
    # 6️⃣ Save JSON
    # ------------------------------------------------------------
    with open(json_path, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"[SUCCESS] COCO file updated at {json_path}")