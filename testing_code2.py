import shutil, os

train_images = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\combined\train\images"
train_labels = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\combined\train\labels"
val_images = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\combined\val\images"
val_labels = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\combined\val\labels"

os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# copying  20% of files from train to val
train_files = os.listdir(train_images)
sample_size = max(1, int(0.2 * len(train_files)))

for f in train_files[:sample_size]:
    shutil.copy(os.path.join(train_images, f), os.path.join(val_images, f))

for f in os.listdir(train_labels)[:sample_size]:
    shutil.copy(os.path.join(train_labels, f), os.path.join(val_labels, f))

print(f"✅ Copied {sample_size} images and labels from train to val folder!")
