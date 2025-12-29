from ultralytics import YOLO

# pretrained COCO model
model = YOLO("yolov8n.pt")

# Train on emergency dataset
model.train(
    data=r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\emergency_vehicles\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="emergency_finetune",
    pretrained=True
)
