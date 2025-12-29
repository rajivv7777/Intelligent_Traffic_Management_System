import shutil, os

# Path to your new emergency vehicle dataset
new_data = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\emergency_vehicles"
combined = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\combined"

for sub in ["train", "valid"]:
    for t in ["images", "labels"]:
        os.makedirs(os.path.join(combined, sub, t), exist_ok=True)
        src = os.path.join(new_data, sub, t)
        if not os.path.exists(src):
            print(f"⚠️ Skipping missing folder: {src}")
            continue
        for f in os.listdir(src):
            shutil.copy(os.path.join(src, f), os.path.join(combined, sub, t))

print("✅ Dataset combined successfully at:", combined)
