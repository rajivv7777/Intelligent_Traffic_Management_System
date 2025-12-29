import os
import shutil

BASE = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\emergency_vehicle"

# Folders
IMG_TRAIN = os.path.join(BASE, "images", "train")
IMG_VAL = os.path.join(BASE, "images", "val")
LBL_TRAIN = os.path.join(BASE, "labels", "train")
LBL_VAL = os.path.join(BASE, "labels", "val")

os.makedirs(IMG_TRAIN, exist_ok=True)
os.makedirs(IMG_VAL, exist_ok=True)
os.makedirs(LBL_TRAIN, exist_ok=True)
os.makedirs(LBL_VAL, exist_ok=True)

# Loop over extracted Roboflow folders
for folder in os.listdir(BASE):
    path = os.path.join(BASE, folder)
    if os.path.isdir(path) and folder.endswith(".yolov8"):  # Roboflow format folder
        print("Processing:", folder)

        # Move images
        if os.path.exists(os.path.join(path, "train")):
            for f in os.listdir(os.path.join(path, "train", "images")):
                shutil.move(os.path.join(path, "train", "images", f), IMG_TRAIN)
            for f in os.listdir(os.path.join(path, "train", "labels")):
                shutil.move(os.path.join(path, "train", "labels", f), LBL_TRAIN)

        if os.path.exists(os.path.join(path, "valid")):
            for f in os.listdir(os.path.join(path, "valid", "images")):
                shutil.move(os.path.join(path, "valid", "images", f), IMG_VAL)
            for f in os.listdir(os.path.join(path, "valid", "labels")):
                shutil.move(os.path.join(path, "valid", "labels", f), LBL_VAL)

print("Dataset structure fixed ✔")
