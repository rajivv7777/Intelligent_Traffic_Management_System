from ultralytics import YOLO

# Path to data.yaml
DATA_YAML = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\emergency_vehicle\emergency_vehicle.yaml"

# Load pretrained YOLO (smallest)
model = YOLO("yolov8n.pt")

# Train
model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    name="emergency_vehicle_training"
)

print("\nTraining completed!")
print("Best model saved at: runs/detect/emergency_vehicle_training/weights/best.pt")
