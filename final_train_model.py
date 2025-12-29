from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\datasets\combined\data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    name="combined_animal_emergency",
    pretrained=True
)
