import cv2
import time
from ultralytics import YOLO
import socket


CAMERA_INDEX = 0                 
CONFIDENCE_THRESHOLD = 0.50

# Current Communication method (will be replaced)
PC2_IP = '10.60.24.219' 
PORT = 65432

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((PC2_IP, PORT))

#--------------------
# INITIALIZATION
#--------------------
print("Loading YOLO11")
model = YOLO('best.pt')  

print("Starting Camera Feed...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open camera. Try changing CAMERA_INDEX to 1.")

#--------------------
# VISION LOOP
#--------------------
while True:
    ret, frame = cap.read()
    if not ret: break

    # Square Crop Math (Prevents Distortion)
    h, w, _ = frame.shape
    size = min(h, w)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    square_frame = frame[y1:y1+size, x1:x1+size]
    final_frame = cv2.resize(square_frame, (640, 640))

    # Inference
    results = model(final_frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            
            if confidence > CONFIDENCE_THRESHOLD:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Draw Visuals
                x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])
                cv2.rectangle(final_frame, (x1_box, y1_box), (x2_box, y2_box), (0, 255, 0), 2)
                cv2.putText(final_frame, f"{class_name} {confidence:.2f}", (x1_box, y1_box - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # --- ARDUINO BYPASS: PRINT TO CONSOLE INSTEAD ---
                print(f"Target Acquired: {class_name.upper()} (Confidence: {confidence * 100:.1f}%)")
                client_socket.sendall((f"Target Acquired: {class_name.upper()} (Confidence: {confidence * 100:.1f}%)" + '\n').encode('utf-8'))

    cv2.imshow("M&M Sorter Vision Test", final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("System Offline.")