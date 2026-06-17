import cv2
from ultralytics import YOLO
import serial
import time
import numpy as np

# CONFIGURATION
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.60
SERIAL_PORT = 'COM5'
BAUD_RATE = 9600
MIN_DETECTION_FRAMES = 8
EMPTY_FRAMES_TO_RESET = 8
MAX_JITTER_FRAMES = 15
STABILITY_THRESHOLD = 200
MIN_STABLE_FRAMES = 5

print("Loading YOLO11...")
model = YOLO('best.pt')

print("Connecting to Arduino...")
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print("Arduino connected.")

print("Starting Camera Feed...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open camera.")

# STATE
state = 'READY'
consecutive_detections = 0
consecutive_empty_frames = 0
jitter_counter = 0

# Stability tracking
last_center = None
stable_frame_count = 0
object_is_stable = False

# HELPERS
def get_center(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def center_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# VISION LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    size = min(h, w)
    cx = (w - size) // 2
    cy = (h - size) // 2
    final_frame = cv2.resize(frame[cy:cy+size, cx:cx+size], (640, 640))

    results = model(final_frame, verbose=False)

    best_detection = None
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > CONFIDENCE_THRESHOLD:
                if best_detection is None or confidence > best_detection[1]:
                    class_name = model.names[int(box.cls[0])]
                    best_detection = (class_name, confidence, box.xyxy[0])

    # STABILITY CHECK
    if best_detection and state == 'READY':
        current_center = get_center(best_detection[2])

        if last_center is not None:
            dist = center_distance(current_center, last_center)
            if dist < STABILITY_THRESHOLD:
                stable_frame_count += 1
            else:
                stable_frame_count = 0
                object_is_stable = False

        if stable_frame_count >= MIN_STABLE_FRAMES:
            object_is_stable = True

        last_center = current_center

    # STATE LOGIC
    if state == 'READY':
        if best_detection and object_is_stable:
            # Stable detection
            jitter_counter = 0
            consecutive_detections += 1
            if consecutive_detections >= MIN_DETECTION_FRAMES:
                class_name = best_detection[0]
                arduino.write(f"{class_name.upper()}\n".encode('utf-8'))
                print(f"[SENT] {class_name.upper()}")
                state = 'WAITING_EMPTY'
                consecutive_detections = 0
                consecutive_empty_frames = 0 

        elif best_detection and not object_is_stable:
            # Detected but not yet stable
            jitter_counter = 0

        else:
            # Nothing detected
            jitter_counter += 1
            if jitter_counter > MAX_JITTER_FRAMES:
                # Reset
                consecutive_detections = 0
                jitter_counter = 0
                last_center = None
                stable_frame_count = 0
                object_is_stable = False

    elif state == 'WAITING_EMPTY':
        if best_detection is None:
            consecutive_empty_frames += 1
            if consecutive_empty_frames >= EMPTY_FRAMES_TO_RESET:
                state = 'READY'
                consecutive_empty_frames = 0
                last_center = None
                stable_frame_count = 0
                object_is_stable = False
                print("[STATE] Zone clear. Ready for next object.")
        else:
            consecutive_empty_frames = 0

    # Visuals
    if best_detection:
        class_name, confidence, coords = best_detection
        x1_b, y1_b, x2_b, y2_b = map(int, coords)
        box_color = (0, 255, 0) if object_is_stable else (0, 165, 255)
        cv2.rectangle(final_frame, (x1_b, y1_b), (x2_b, y2_b), box_color, 2)
        cv2.putText(final_frame, f"{class_name} {confidence:.2f}",
                    (x1_b, y1_b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    stability_str = f"Stable: {stable_frame_count}/{MIN_STABLE_FRAMES}"
    hud = f"State: {state} | Streak: {consecutive_detections}/{MIN_DETECTION_FRAMES} | {stability_str}"
    cv2.putText(final_frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("M&M Sorter", final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEANUP
cap.release()
arduino.close()
cv2.destroyAllWindows()
print("System Offline.")
