from ultralytics import YOLO
import torch

model = YOLO('yolo11n.pt')

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU Found.")

    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        device=0,
        patience=15,
    )
