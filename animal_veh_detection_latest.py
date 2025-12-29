import cv2
from ultralytics import YOLO
from playsound import playsound
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

# --- Configuration ---
ANIMAL_MODEL_PATH = "yolov8n.pt"
VEHICLE_MODEL_PATH = r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\runs\detect\emergency_vehicle_training5\weights\best.pt"

SENDER_EMAIL = "599rajiv.2020@gmail.com"
SENDER_PASSWORD = "lyaz atwv vorn rwuv"
RECEIVER_EMAIL = "rajivv7777@gmail.com"

os.makedirs("detections/animals", exist_ok=True)
os.makedirs("detections/vehicles", exist_ok=True)
os.makedirs("logs", exist_ok=True)

alerted_ids = set()
CONF_THRESHOLD = 0.70

# --- Helpers ---
def play_alert_sound():
    threading.Thread(target=playsound, args=("alert.mp3",), daemon=True).start()

def log_event(text):
    with open("logs/detection_log.txt", "a") as f:
        f.write(text + "\n")

def send_email_alert(class_name, confidence, track_id, kind="Detection"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"{kind} - {class_name} (ID {track_id})"
    body = f"{class_name} detected at {timestamp} with confidence {confidence:.2f}. Tracking ID: {track_id}"

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print(f"[EMAIL] Sent: {subject}")
    except Exception as e:
        print("[EMAIL ERROR]", e)

# --- Load Models ---
print("Loading models...")
model = YOLO(ANIMAL_MODEL_PATH)
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
print("Models loaded.")

# ======================================================
#           STEP 1: PRINT CLASS NAME LIST
# ======================================================
print("\n================ CLASS NAME CHECK ================")
print("Animal model class names:", model.names)
print("Vehicle model class names:", vehicle_model.names)
print("===================================================\n")

# DO NOT CHANGE ANYTHING BELOW UNTIL WE SEE THE OUTPUT
# (We will fix incorrect vehicle classes after this step)

vehicle_classes = ['Ambulance', 'Police', 'Firebrigade', 'Army_vehicle']
animal_classes = ['cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']

# --- Camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open camera")
    exit()

print("Starting detection. Press 'x' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results_animal = model.track(frame, persist=True)[0]
    results_vehicle = vehicle_model.track(frame, persist=True)[0]

    animal_count = sum(1 for box in results_animal.boxes if model.names[int(box.cls[0])] in animal_classes)
    vehicle_count = sum(1 for box in results_vehicle.boxes if vehicle_model.names[int(box.cls[0])] in vehicle_classes)

    cv2.putText(frame, f"Animals: {animal_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # ===================== ANIMAL DETECTION =====================
    for box in results_animal.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"AID{track_id}:{class_name} {conf:.2f}"

        if class_name in animal_classes and track_id != -1 and conf >= CONF_THRESHOLD:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 20, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 20, 0), 2)

            aid = f"A-{track_id}"
            if aid not in alerted_ids:
                play_alert_sound()
                send_email_alert(class_name, conf, track_id, kind="Animal")
                log_event(f"[ANIMAL] {class_name}, Conf: {conf:.2f}, ID: {track_id}, Time: {datetime.now()}")
                alerted_ids.add(aid)
                fname = f"detections/animals/{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fname, frame)
                print("[ANIMAL] Saved", fname)

    # ===================== VEHICLE DETECTION =====================
    for box in results_vehicle.boxes:
        cls_id = int(box.cls[0])
        class_name = vehicle_model.names[cls_id]
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"VID{track_id}:{class_name} {conf:.2f}"

        if class_name in vehicle_classes and track_id != -1 and conf >= CONF_THRESHOLD:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)

            vid = f"V-{track_id}"
            if vid not in alerted_ids:
                play_alert_sound()
                send_email_alert(class_name, conf, track_id, kind="Vehicle")
                log_event(f"[VEHICLE] {class_name}, Conf: {conf:.2f}, ID: {track_id}, Time: {datetime.now()}")
                alerted_ids.add(vid)
                fname = f"detections/vehicles/{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fname, frame)
                print("[VEHICLE] Saved", fname)

    cv2.imshow("Animal & Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()

