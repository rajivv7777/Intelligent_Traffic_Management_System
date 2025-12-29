import os, shutil

animal_data = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\animals"
emergency_data = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\emergency_vehicles"
combined = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\combined"

# Creating combined structure
for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
    os.makedirs(os.path.join(combined, sub), exist_ok=True)

# Helper function to copy files
def copy_data(src, dest):
    for root, _, files in os.walk(src):
        for f in files:
            shutil.copy(os.path.join(root, f), os.path.join(dest, f))

# Copy animal dataset
copy_data(os.path.join(animal_data, "train/images"), os.path.join(combined, "train/images"))
copy_data(os.path.join(animal_data, "train/labels"), os.path.join(combined, "train/labels"))
copy_data(os.path.join(animal_data, "val/images"), os.path.join(combined, "val/images"))
copy_data(os.path.join(animal_data, "val/labels"), os.path.join(combined, "val/labels"))

# Copy emergency dataset
copy_data(os.path.join(emergency_data, "train/images"), os.path.join(combined, "train/images"))
copy_data(os.path.join(emergency_data, "train/labels"), os.path.join(combined, "train/labels"))
copy_data(os.path.join(emergency_data, "val/images"), os.path.join(combined, "val/images"))
copy_data(os.path.join(emergency_data, "val/labels"), os.path.join(combined, "val/labels"))

print("✅ Combined dataset created successfully at:", combined)
