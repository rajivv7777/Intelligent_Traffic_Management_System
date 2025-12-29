import cv2
from ultralytics import YOLO
from playsound import playsound
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
import time

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
emergency_alerted_ids = set()
CONF_THRESHOLD = 0.70

# ---- Traffic light state (global) ----
# We'll implement two approaches: LEFT and RIGHT (determined by centroid x)
# current_phase indicates which approach currently has GREEN
current_phase = "LEFT"   # or "RIGHT"
phase_start_time = time.time()

# Adaptive timing params (seconds)
BASE_GREEN = 5           # minimum base green
K_PER_VEHICLE = 2.0      # additional seconds per detected vehicle
MIN_GREEN = 5
MAX_GREEN = 30

# Emergency preemption state
ev_detected_flag = False
last_detect_time = 0
ev_preempt_hold = 8      # hold green for this many seconds after preemption
ev_preempt_end_time = 0
ev_target_phase = None
screen_message = ""

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

print("\n================ CLASS NAME CHECK ================")
print("Animal model class names:", model.names)
print("Vehicle model class names:", vehicle_model.names)
print("===================================================\n")

vehicle_classes = ['Ambulance', 'Police', 'Firebrigade', 'Army_vehicle']
animal_classes = ['cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']

# --- Camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open camera")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Starting detection. Press 'x' to exit.")

def compute_adaptive_green(count):
    """Compute green duration based on detected vehicle count."""
    duration = BASE_GREEN + K_PER_VEHICLE * count
    duration = max(MIN_GREEN, min(MAX_GREEN, duration))
    return duration

def draw_traffic_light_overlay(img, current_phase, time_left):
    """Draw a simple two-approach traffic light and countdown on the frame."""
    # top-left box for LEFT approach
    cv2.rectangle(img, (10, 80), (120, 220), (50, 50, 50), -1)
    # top-right box for RIGHT approach
    cv2.rectangle(img, (frame_width - 130, 80), (frame_width - 20,200), (50, 50, 50), -1)

    # LEFT light
    left_color = (0, 255, 0) if current_phase == "LEFT" else (0, 0, 255)
    cv2.circle(img, (65, 120), 20, left_color, -1)
    cv2.putText(img, "LEFT", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # RIGHT light
    right_color = (0, 255, 0) if current_phase == "RIGHT" else (0, 0, 255)
    cv2.circle(img, (frame_width - 75, 120), 20, right_color, -1)
    cv2.putText(img, "RIGHT", (frame_width - 110, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Countdown
    cv2.putText(img, f"Time left: {int(time_left)}s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results_animal = model.track(frame, persist=True)[0]
    results_vehicle = vehicle_model.track(frame, persist=True)[0]

    # Basic counts for display (all animals and emergency-related vehicles)
    animal_count = sum(1 for box in results_animal.boxes if model.names[int(box.cls[0])] in animal_classes)
    total_vehicle_count = sum(1 for box in results_vehicle.boxes if vehicle_model.names[int(box.cls[0])] in vehicle_classes or True)

    cv2.putText(frame, f"Animals: {animal_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Vehicles (total boxes): {len(results_vehicle.boxes)}", (10, 60),
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

    # ===================== EMERGENCY VEHICLE DETECTION =====================
    emergency_detected = False
    confidence_high = False
    detected_track_id = -1
    detected_class = ""
    detected_centroid_x = None

    # We'll also build approach-wise vehicle counts (LEFT / RIGHT)
    counts = {"LEFT": 0, "RIGHT": 0}

    for box in results_vehicle.boxes:
        cls_id = int(box.cls[0])
        # If model.names doesn't map because of custom classes, fallback safe
        class_name = vehicle_model.names.get(cls_id, str(cls_id)) if isinstance(vehicle_model.names, dict) else vehicle_model.names[cls_id]
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)

        # Determine approach by centroid x
        approach = "LEFT" if cx < frame_width // 2 else "RIGHT"
        counts[approach] += 1

        # Draw bounding boxes for vehicles (for debug/visual)
        label = f"VID{track_id}:{class_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 2)

        # Emergency class check
        if class_name in vehicle_classes and track_id != -1 and conf >= CONF_THRESHOLD:
            emergency_detected = True
            detected_track_id = track_id
            detected_class = class_name
            detected_centroid_x = cx
            if conf >= 0.85:
                confidence_high = True

    # ----- Handle emergency vehicle logic (no bounding box) -----
    if emergency_detected and confidence_high:
        eid = f"E-{detected_track_id}"

        # decide which approach the EV is on
        ev_approach = "LEFT" if detected_centroid_x is not None and detected_centroid_x < frame_width // 2 else "RIGHT"
        screen_message = f"🚨 EV ({detected_class}) detected on {ev_approach} — switching to GREEN"

        if eid not in emergency_alerted_ids:
            print(f"\n🚨 Emergency vehicle detected — {detected_class} ID {detected_track_id} on {ev_approach}")
            log_event(f"[EMERGENCY] {detected_class}, Conf > 0.85, ID: {detected_track_id}, Approach: {ev_approach}, Time: {datetime.now()}")
            emergency_alerted_ids.add(eid)

        # Preempt: immediately switch phase to EV approach
        current_phase = ev_approach
        ev_detected_flag = True
        last_detect_time = time.time()
        ev_preempt_end_time = time.time() + ev_preempt_hold
        ev_target_phase = ev_approach

    elif emergency_detected and not confidence_high:
        screen_message = "Emergency detected but confidence < 85%"
        print("Emergency vehicle detected but confidence < 85% — Light NOT changing")

    # -------- Adaptive Traffic Signal Logic --------
    # Compute dynamic green durations for each approach based on counts
    green_left = compute_adaptive_green(counts["LEFT"])
    green_right = compute_adaptive_green(counts["RIGHT"])

    # Time elapsed in current phase
    elapsed_phase = time.time() - phase_start_time

    # If we're in emergency preempt hold window, keep the EV approach green until hold ends
    if ev_detected_flag and time.time() < ev_preempt_end_time:
        # keep current_phase as EV approach; reset phase_start_time so countdown uses the preempt hold time
        phase_remaining = ev_preempt_end_time - time.time()
        current_green_duration = ev_preempt_end_time - phase_start_time if phase_start_time < ev_preempt_end_time else ev_preempt_hold
        time_left = max(0, ev_preempt_end_time - time.time())
        # keep ev_detected_flag True until preempt window expires
    else:
        # If the preempt window expired earlier, clear flag (we'll resume normal adaptive control)
        if ev_detected_flag and time.time() >= ev_preempt_end_time:
            ev_detected_flag = False
            ev_target_phase = None
            # start new phase timer to avoid immediate switch jitter
            phase_start_time = time.time()
            elapsed_phase = 0

        # Normal adaptive switching based on computed green durations
        if current_phase == "LEFT":
            current_green_duration = green_left
            time_left = max(0, current_green_duration - elapsed_phase)
            if elapsed_phase >= current_green_duration:
                # switch to RIGHT
                current_phase = "RIGHT"
                phase_start_time = time.time()
                current_green_duration = green_right
                time_left = current_green_duration
        else:  # RIGHT
            current_green_duration = green_right
            time_left = max(0, current_green_duration - elapsed_phase)
            if elapsed_phase >= current_green_duration:
                # switch to LEFT
                current_phase = "LEFT"
                phase_start_time = time.time()
                current_green_duration = green_left
                time_left = current_green_duration

    # Draw a traffic-light overlay and countdown
    draw_traffic_light_overlay(frame, current_phase, time_left)

    # -------- Draw screen message ONLY after strong confirmed emergency detection --------
    if ev_detected_flag and confidence_high and screen_message != "":
        # show message near bottom
        cv2.putText(frame, screen_message, (10, frame_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Debug: show approach counts
    cv2.putText(frame, f"Count LEFT: {counts['LEFT']}", (10, frame_height - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(frame, f"Count RIGHT: {counts['RIGHT']}", (10, frame_height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    # =====================================================
    cv2.imshow("Animal & Vehicle Detection with Adaptive Signal", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
