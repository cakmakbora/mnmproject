from ultralytics import YOLO
import torch

model = YOLO('yolo11n.pt') 

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"Success! Using GPU: {torch.cuda.get_device_name(0)}")
    
    results = model.train(data='data.yaml', epochs=50, imgsz=640, device=0)
