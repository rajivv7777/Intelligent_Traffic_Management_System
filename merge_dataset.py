#STEP 3
import os
import shutil

# ---------------------------------------------------
# CONFIG — Change only if your folders change
# ---------------------------------------------------

BASE_DATASET = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\emergency_vehicle"

# Extracted dataset folders (inside emergency_vehicle folder)
EXTRACTED_FOLDERS = [
    "Ambulance.v1i.yolov8",
    "Police_car.v1i.yolov8",
    "Firebrigade.v1i.yolov8",
    "Army_vehicle.v1i.yolov8"
]

# Prefix for each class
PREFIX = {
    "Ambulance.v1i.yolov8": "amb",
    "Police_car.v1i.yolov8": "pol",
    "Firebrigade.v1i.yolov8": "fire",
    "Army_vehicle.v1i.yolov8": "army"
}

# ---------------------------------------------------
# CREATE MERGED DATASET
# ---------------------------------------------------

def merge_datasets():

    images_final = os.path.join(BASE_DATASET, "images")
    labels_final = os.path.join(BASE_DATASET, "labels")

    os.makedirs(images_final, exist_ok=True)
    os.makedirs(labels_final, exist_ok=True)

    counter = 1

    for folder in EXTRACTED_FOLDERS:
        current_path = os.path.join(BASE_DATASET, folder)

        img_path = os.path.join(current_path, "train", "images")  # Roboflow format
        lbl_path = os.path.join(current_path, "train", "labels")

        if not os.path.exists(img_path):
            print(f"[WARNING] Missing images folder for: {folder}")
            continue

        print(f"\n[PROCESSING] {folder}")

        for img_file in os.listdir(img_path):

            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            # new consistent name
            new_img_name = f"{PREFIX[folder]}_{counter}.jpg"
            new_lbl_name = f"{PREFIX[folder]}_{counter}.txt"

            # copy image
            shutil.copy(
                os.path.join(img_path, img_file),
                os.path.join(images_final, new_img_name)
            )

            # copy label
            lbl_file = img_file.rsplit(".", 1)[0] + ".txt"
            lbl_src = os.path.join(lbl_path, lbl_file)

            if os.path.exists(lbl_src):
                shutil.copy(
                    lbl_src,
                    os.path.join(labels_final, new_lbl_name)
                )
            else:
                print(f"[WARNING] Label not found: {lbl_src}")

            counter += 1

    print("\n--------------------------------------------")
    print(" MERGING COMPLETE ✔ ")
    print("--------------------------------------------")
    print(f"Total Samples: {counter-1}")
    print(f"Merged Images: {images_final}")
    print(f"Merged Labels: {labels_final}")
    print("--------------------------------------------")


# run function
merge_datasets()
