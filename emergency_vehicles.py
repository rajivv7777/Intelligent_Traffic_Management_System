import os

base = "datasets/emergency_vehicles"
for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
    os.makedirs(os.path.join(base, sub), exist_ok=True)

print("✅ Folder structure created successfully!")
