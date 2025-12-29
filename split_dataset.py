#STEP 4 than step 6 emergency_vehicle.yaml
import os
import shutil
import random

# ---------------------------------------------------
# CONFIG - MODIFY ONLY IF REQUIRED
# ---------------------------------------------------
BASE_DATASET = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\emergency_vehicle"

train_ratio = 0.70   # 70% images
val_ratio   = 0.20   # 20% images
test_ratio  = 0.10   # 10% images

# ---------------------------------------------------
# CREATE REQUIRED FOLDERS
# ---------------------------------------------------
def create_split_folders():

    for folder in ["images", "labels"]:
        for split in ["train", "val", "test"]:
            path = os.path.join(BASE_DATASET, folder, split)
            os.makedirs(path, exist_ok=True)

# ---------------------------------------------------
# MAIN SPLIT FUNCTION
# ---------------------------------------------------
def split_dataset():

    img_dir = os.path.join(BASE_DATASET, "images")
    lbl_dir = os.path.join(BASE_DATASET, "labels")

    images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    print(f"\nTotal images: {total}")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}\n")

    # Function to move files
    def move_files(file_list, split):

        for img in file_list:
            label = img.rsplit(".", 1)[0] + ".txt"

            # Move image
            shutil.move(
                os.path.join(img_dir, img),
                os.path.join(BASE_DATASET, "images", split, img)
            )

            # Move label
            shutil.move(
                os.path.join(lbl_dir, label),
                os.path.join(BASE_DATASET, "labels", split, label)
            )

    move_files(train_files, "train")
    move_files(val_files, "val")
    move_files(test_files, "test")

    print("-------------------------------------------")
    print(" Dataset Split Completed ✔")
    print("-------------------------------------------")
    print("Train Folder:", train_end)
    print("Val Folder:", len(val_files))
    print("Test Folder:", len(test_files))
    print("-------------------------------------------")

# ---------------------------------------------------
# RUN
# ---------------------------------------------------
create_split_folders()
split_dataset()
