import cv2
from ultralytics import YOLO

from playsound import playsound
import threading

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from datetime import datetime

import os

from unicodedata import category

os.makedirs("detections/animals", exist_ok=True)# Create a folder to save detected images
os.makedirs("detections/vehicles", exist_ok=True)
alerted_ids={}
def play_alert_sound():
    threading.Thread(target=playsound, args=("alert.mp3",),daemon=True).start()

def send_email_alert(class_name, confidence, track_id):    #sms alert
    sender_email = "599rajiv.2020@gmail.com"
    sender_password = "lyaz atwv vorn rwuv"
    receiver_email = "rajivv7777@gmail.com"

    timestamp =datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    subject = f"{category} Detected: {class_name} (ID {track_id})"
    body = (
        f"A {category.lower()} ({class_name}) was detected at {timestamp} "
        f"with {confidence:.2f} confidence.\nTracking ID: {track_id}"
    )
    #subject = f"Animal Detected: {class_name} (ID {track_id})"
    #body = f"An animal ({class_name}) was detected at {timestamp} with {confidence:.2f} confidence.\nTracking ID: {track_id}"

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
        print(f"✅ Email alert sent for {class_name} (ID {track_id}) at {timestamp}")
    except Exception as e:
        print(f"❌ Error sending email: {e}")

model = YOLO("yolov8n.pt")
vehicle_model=YOLO(r"C:\RAJIV\PROJECT\Intelligence Traffic Management System\runs\detect\emergency_vehicle_training2\weights\best.pt")

#camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error - could not open camera !!!")
    exit()

animal_classes = [ 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe','buffalo', 'camel', 'donkey', 'pig', 'tiger', 'lion',
    'leopard', 'monkey', 'fox', 'bear', 'zebra', 'giraffe',
    'rabbit', 'hen', 'peacock', 'bull', 'wolf', 'rhinoceros']
vehicle_classes= [ 'ambulance', 'fire_truck', 'police_car', 'army']
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)[0]   #Running YOLO v8 inference on the frame
    #count detect animals
    animal_count = sum(1 for box in results.boxes if model.names[int(box.cls[0])] in animal_classes)
    vehicle_count= sum(1 for box in results.boxes if model.names[int(box.cls[0])] in vehicle_classes)

    cv2.putText(frame, f"Animals: {animal_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = box.conf[0]
        track_id = int(box.id[0]) if box.id is not None else -1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"ID{track_id}:{class_name} {conf:.2f}"

         #FOR ANIMAL DETECTION
        if class_name in animal_classes and track_id !=-1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,20,0),1) #x1, y1) and (x2, y2): top-left and bottom-right corner coordinates of the box,(0, 255, 0)=RGB,1: the thickness of the rectangle.
            cv2.putText(frame, label, (x1, y1 - 10),#(x1, y1 - 10): the position to draw the text (above the bounding box)
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # If this ID hasn't been alerted yet → send mail + sound
            if track_id not in alerted_ids:
                play_alert_sound()
                send_email_alert(class_name, conf, track_id)
                # Mark this ID as alerted
                alerted_ids[track_id] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # save detection image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detections/animals/{class_name}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

                # For Important Vehicle detection
            elif class_name in vehicle_classes and track_id != -1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)

                if track_id not in alerted_ids:
                    play_alert_sound()
                    send_email_alert("Important Vehicle", class_name, conf, track_id)
                    alerted_ids[track_id] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detections/vehicles/{class_name}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)


           # with open("detection_log.txt", "a") as f:   #detection time         # sending mail after detection
            #    f.write(f"{alerted_ids[track_id]} - {class_name} (ID {track_id}) detected with {conf:.2f} confidence\n")

            #current_time= time.time()
            #if current_time-last_alert_time>ALERT_COOLDOWN: # cooldown
             #   send_email_alert(class_name,conf)
              #  last_alert_time=current_time



    # Show result
    cv2.imshow("Animal & Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
