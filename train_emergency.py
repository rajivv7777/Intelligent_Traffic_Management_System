from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Train on your emergency vehicle dataset
model.train(
    data=r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\combined\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="emergency_vehicle_training"
)
