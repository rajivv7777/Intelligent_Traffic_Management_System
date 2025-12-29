import cv2
from ultralytics import YOLO
from playsound import playsound
import threading

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

# Create folders
os.makedirs("detections/animals", exist_ok=True)
os.makedirs("detections/vehicles", exist_ok=True)

alerted_ids = {}

# PLAY SOUND
def play_alert_sound():
    threading.Thread(target=playsound, args=("alert.mp3",), daemon=True).start()

# EMAIL ALERT
def send_email_alert(class_name, confidence, track_id):
    sender_email = "599rajiv.2020@gmail.com"
    sender_password = "lyaz atwv vorn rwuv"
    receiver_email = "rajivv7777@gmail.com"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"Detected: {class_name} (ID {track_id})"
    body = (
        f"{class_name} detected at {timestamp} "
        f"with confidence {confidence:.2f}.\nTracking ID: {track_id}"
    )

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"Email alert sent for {class_name} (ID {track_id})")
    except Exception as e:
        print("Error sending email:", e)

# LOAD MODELS
model = YOLO("yolov8n.pt")
vehicle_model = YOLO(r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\runs\detect\emergency_vehicle_training2\weights\best.pt")

# CAMERA
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera Error!")
    exit()

# CLASS LIST
animal_classes = ['cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']
vehicle_classes = ['ambulance','fire_truck','police_car','army']

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)[0]

    # COUNTS
    animal_count = sum(1 for box in results.boxes
                       if model.names[int(box.cls[0])] in animal_classes)

    vehicle_count = sum(1 for box in results.boxes
                        if model.names[int(box.cls[0])] in vehicle_classes)

    cv2.putText(frame, f"Animals: {animal_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # PROCESS EACH BOX
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"ID{track_id}:{class_name} {conf:.2f}"

        # ------------------- ANIMAL DETECTION -------------------
        if class_name in animal_classes and track_id != -1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 20, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 20, 0), 2)

            if track_id not in alerted_ids:
                play_alert_sound()
                send_email_alert(class_name, conf, track_id)
                alerted_ids[track_id] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                filename = f"detections/animals/{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)

        # ---------------- VEHICLE DETECTION (FIXED BLOCK) ----------------
        if class_name in vehicle_classes and track_id != -1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)

            if track_id not in alerted_ids:
                play_alert_sound()
                send_email_alert(class_name, conf, track_id)
                alerted_ids[track_id] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                filename = f"detections/vehicles/{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)

    cv2.imshow("Animal & Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
