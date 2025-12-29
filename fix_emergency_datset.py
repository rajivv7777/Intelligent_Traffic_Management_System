import os
import shutil

BASE = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\emergency_vehicle"

def ensure(path):
    os.makedirs(path, exist_ok=True)

# YOLO structure
IMG_TRAIN = os.path.join(BASE, "images", "train")
IMG_VAL   = os.path.join(BASE, "images", "val")
IMG_TEST  = os.path.join(BASE, "images", "test")

LBL_TRAIN = os.path.join(BASE, "labels", "train")
LBL_VAL   = os.path.join(BASE, "labels", "val")
LBL_TEST  = os.path.join(BASE, "labels", "test")

# Create required YOLO folders
ensure(IMG_TRAIN); ensure(IMG_VAL); ensure(IMG_TEST)
ensure(LBL_TRAIN); ensure(LBL_VAL); ensure(LBL_TEST)

def move_split(split):
    print(f"\nProcessing: {split}")
    split_path = os.path.join(BASE, split)

    img_src = os.path.join(split_path, "images")
    lbl_src = os.path.join(split_path, "labels")

    dst_img = os.path.join(BASE, "images", split)
    dst_lbl = os.path.join(BASE, "labels", split)

    if not os.path.exists(img_src):
        print(f"❌ No images folder in: {split_path}")
        return

    for f in os.listdir(img_src):
        shutil.move(os.path.join(img_src, f), dst_img)

    for f in os.listdir(lbl_src):
        shutil.move(os.path.join(lbl_src, f), dst_lbl)

    print(f"✔ Moved: {split}")

# Move train/val/test from Roboflow format
move_split("train")
move_split("valid")
move_split("test")

print("\n🎉 Dataset converted to correct YOLO format successfully!")
