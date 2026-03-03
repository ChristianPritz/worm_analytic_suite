#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 12:35:27 2026

@author: wormulon
"""

from coco_to_yolo import *
from measurements import *
from annotation_tool_v10 import AnnotationTool
# Merge json paths 


#joining all the subdatasets into a single COCO dataset 

baseDir = '/media/my_device/worm analytic suite/COCO dataset/'

organize_dataset(baseDir)

COCO_images = '/media/my_device/worm analytic suite/COCO dataset/images'
COCO_points = '/media/my_device/worm analytic suite/COCO dataset/labels/annotations_points.csv'
COCO_json = '/media/my_device/worm analytic suite/COCO dataset/labels/annotations_areas.json'
class_csv = '/home/wormulon/models/worm_analytic_suite/class_colors.csv'

# merge the datasets
settings = {"rim_score_cutoff":0.05,"debug":False,"use_classifier":True}
collect_annotations(settings,
    anno_dir='/media/my_device/worm analytic suite/COCO dataset/labels', # this is where there raw annoations go
    image_dir='/media/my_device/worm analytic suite/COCO dataset/images', # this is where the images are sitting
    color_csv=class_csv,
    output_dir='/media/my_device/worm analytic suite/COCO dataset/merged_labels' #This is where the joined labels go 
)


COCO_points = '/media/my_device/worm analytic suite/COCO dataset/merged_labels/annotations_points.csv'
COCO_json = '/media/my_device/worm analytic suite/COCO dataset/merged_labels/annotations_areas.json'

tool = AnnotationTool(COCO_images,COCO_points,COCO_json,class_csv)





json_file = "" 

train_json = ""
val_json =  ""

base_Dir_yolo_files = ""

# partition the COCO data set into training and validation data.
coco_train = '/media/my_device/worm analytic suite/COCO dataset/merged_labels/coco_train.json'
coco_val = '/media/my_device/worm analytic suite/COCO dataset/merged_labels/coco_val.json'

COCO_base = '/media/my_device/worm analytic suite/COCO dataset/merged_labels/annotations_areas.json'

split_coco_annotations(
    COCO_base,
    coco_train,
    coco_val,
    train_ratio=0.99)




# clean the json files. 

inspect_and_verify_coco(
    coco_train,
    delete_invalid=False,
    shrink_keypoints=True)

inspect_and_verify_coco(
    coco_val,
    delete_invalid=False,
    shrink_keypoints=True)



coco_train_clean = "/media/my_device/worm analytic suite/COCO dataset/merged_labels/coco_train_verified.json"
coco_val_clean = "/media/my_device/worm analytic suite/COCO dataset/merged_labels/coco_val_verified.json"
# converting the COCO data set into a YOLO data set

yolo_base_path = "/media/my_device/worm analytic suite/YOLO dataset/"
baseDir_img = "/media/my_device/worm analytic suite/COCO dataset/images"
baseDir = "/media/my_device/worm analytic suite/YOLO dataset/"
convert_coco_json(
        src_img_dir = baseDir_img, 
        json_path=coco_train_clean,
        images_dir=yolo_base_path + "/images/train",
        output_dir=yolo_base_path + "/labels/train",
        base_dir=baseDir,
        use_segments=True,
        use_keypoints=False,
        label="train_")


convert_coco_json(
        src_img_dir = baseDir_img, 
        json_path=coco_val_clean,
        images_dir=yolo_base_path + "images/val",
        output_dir=yolo_base_path + "labels/val",
        base_dir=baseDir,
        use_segments=True,
        use_keypoints=False,
        label = "val_"
    )




# ACTUAL TRAINING##############################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from coco_to_yolo import *

def show_segmentation(image_path, results, alpha=0.5):
    """
    Display an image with YOLOv8 segmentation results overlayed.
    Automatically resizes masks to match the original image.
    
    Parameters
    ----------
    image_path : str or Path
        Path to the input image.
    results : ultralytics.yolo.engine.results.Results
        The results object returned by model.predict().
    alpha : float
        Transparency for mask overlay (0 = invisible, 1 = opaque).
    """
    # Load the original image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]

    overlay = img.copy()

    # Iterate over masks and classes
    for mask, cls in zip(results.masks.data, results.boxes.cls):
        mask = mask.cpu().numpy().astype(np.uint8)  # binary mask 0/1

        # Resize mask to original image size
        mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        # Create a colored mask
        color = np.random.randint(0, 255, size=3)
        colored_mask = np.zeros_like(img, dtype=np.uint8)
        for c in range(3):
            colored_mask[:,:,c] = mask_resized * color[c]

        # Blend the mask onto the overlay
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)

        # Draw contours
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2)

    # Display the image
    plt.figure(figsize=(10,10))
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

# Path to your dataset YAML
dataset_yaml = "/media/my_device/worm analytic suite/YOLO dataset/train_dataset.yaml"

# Choose a pre-trained YOLOv8 segmentation model (nano, small, medium, large)
# For RTX 4070 Ti, medium should be fine for reasonable speed
#model = YOLO("yolov8m-seg.pt")  # segmentation pre-trained model
model = YOLO("yolov8l-seg.pt") # slower but better(er)

# Train
# model.train(
#     data=dataset_yaml,
#     imgsz=640,            # resize images to 640x640
#     epochs=100,           # number of epochs
#     batch=8,              # adjust batch size based on GPU VRAM
#     device=0,             # GPU ID (0 = first GPU)
#     workers=4,            # number of data loader workers
#     name="worm_segmentation_custom",  # folder to save weights
#     project="YOLO_training",          # project folder
#     exist_ok=True
# )

model.train(
    data=dataset_yaml,
    imgsz=1024, 
    epochs=300,
    batch=4, # increase if enough memory remains...  
    device=0,
    workers=4,
    name="worm_segmentation_custom",
    project="YOLO_training",
    exist_ok=True,

    # 🔬 geometric
    degrees=45,
    translate=0.15,
    scale=0.4,
    shear=2.0,
    fliplr=0.5,
    flipud=0.5,

    # 🎨 color
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,

    # 🧪 optional
    mosaic=0.5,
    mixup=0.0,
    close_mosaic=15, # switches is off in the last 10 epochs. #1.0,          # mosaic augmentation
       # mixup augmentation
)


#manual save
model.save("/media/my_device/worm analytic suite/YOLO dataset/worm_seg_model.pt")

YOLO_training/worm_segmentation_custom/weights/best.pt

#Run inference:

# Load trained model
model = YOLO("/media/my_device/worm analytic suite/YOLO dataset/worm_seg_model.pt")




# Run prediction
image_path = "/media/my_device/worm analytic suite/YOLO dataset/images/val/C_LH_G2_T3_040925rep3_20251105_131348_6.png"
image_path = "/home/wormulon/Downloads/export_LH_G1_image4.png"


model = YOLO("/media/my_device/worm analytic suite/YOLO dataset/worm_seg_model.pt")
json_path =  "/media/my_device/worm analytic suite/prediction labels/to_correct.json"
search_path = "/media/my_device/worm analytic suite/prediction test/*"

#search_path = "/media/my_device/worm analytic suite/COCO dataset/images/*"
import glob

file_list = glob.glob(search_path)
for i in file_list:
    print('###################### instance is #######################')
    print(i)
    results = model.predict(i,show=False)
    yolo_to_coco_adapter(i, results[0], json_path)
    show_segmentation(i, results[0])




# Test if the labels can be edited. 
from coco_to_yolo import *
from measurements import *
from annotation_tool_v11 import AnnotationTool



orig_json = "/media/my_device/worm analytic suite/prediction labels/cleanA.json"


params = {
    "smoothing": {
        "enabled": True,
        "sigma": 1.0
    },
    "interpolation": {
        "mode": "fixed_points",
        "num_points": 100
    }
}

params = {
    "smoothing": {
        "enabled": True,
        "sigma": 1.0
    },
    "interpolation": {
        "mode": "fixed_density",
        "density": 0.05
    }
}

smooth_and_interpolate_coco(orig_json, params)






image_path =  "/media/my_device/worm analytic suite/prediction test/"
COCO_points = '/media/my_device/worm analytic suite/prediction labels/annotations_points.csv'
class_csv = '/home/wormulon/models/worm_analytic_suite/class_colors.csv'
json_path =  "/media/my_device/worm analytic suite/prediction labels/cleanA.json"


tool = AnnotationTool(image_path,COCO_points,orig_json,class_csv)

    

# Example usage:



r = results[0]
class_ids = r.boxes.cls.cpu().numpy().astype(int)


#Converting the Yolo output to our COCO style format


