#STEP 1
import os

# ---------------------------------------------
# CONFIG: Change this to your chosen directory
# ---------------------------------------------
BASE_PATH = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets"  # <-- put your desired full path here

# Dataset Folder Name
DATASET_NAME = "emergency_vehicle"

# Final Classes
CLASSES = [
    "Ambulance",
    "Police",
    "Firebrigade",
    "Army_vehicle"
]

# ----------------------------------------------------
# CREATE DATASET FOLDER STRUCTURE
# ----------------------------------------------------
def create_dataset_structure():
    dataset_path = os.path.join(BASE_PATH, DATASET_NAME)
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    # Create main dataset folder
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    else:
        print(f"[INFO] Folder already exists: {dataset_path}")

    # Create subfolders
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # Create classes file
    classes_file = os.path.join(dataset_path, "classes.txt")
    with open(classes_file, "w") as f:
        for c in CLASSES:
            f.write(f"{c}\n")

    print("\n-------------------------------")
    print(" Dataset Structure Created ✔")
    print("-------------------------------")
    print(f"Base Path: {BASE_PATH}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Images Folder: {images_path}")
    print(f"Labels Folder: {labels_path}")
    print("-------------------------------")

# -------------------------
# Run function
# -------------------------
create_dataset_structure()
