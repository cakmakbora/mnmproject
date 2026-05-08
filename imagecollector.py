import cv2
import os
import time

save_dir = "dataset/images"
os.makedirs(save_dir, exist_ok=True)
cap = cv2.VideoCapture(0)

print("Press 's' to save a square image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # The Square Crop Math
    h, w, _ = frame.shape
    size = min(h, w)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    
    square_frame = frame[y1:y1+size, x1:x1+size]
    final_frame = cv2.resize(square_frame, (640, 640))

    cv2.imshow("Data Collection (640x640)", final_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = os.path.join(save_dir, f"img_{int(time.time())}.jpg")
        cv2.imwrite(filename, final_frame)
        print(f"Saved: {filename}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()