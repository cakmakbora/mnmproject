# M&M Sorter — Setup & Usage Guide

## Requirements

Python 3.8 or higher is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Configuration

Before running any script, open it and adjust the following variables at the top:

| Variable | Description |
|---|---|
| `CAMERA_INDEX` | Camera device index. Try `0` or `1` depending on your setup. |
| `SERIAL_PORT` | Arduino port. Windows: `COM3`, `COM5` etc. Linux: `/dev/ttyUSB0` |
| `CONFIDENCE_THRESHOLD` | Minimum detection confidence to consider a result valid (0.0–1.0) |
| `MIN_DETECTION_FRAMES` | Consecutive frames required to confirm a detection |
| `EMPTY_FRAMES_TO_RESET` | Consecutive empty frames required to re-arm the system |
| `MAX_JITTER_FRAMES` | Flickering frames tolerated without resetting the streak |

---

## Tutorial: Running the Sorter with the Existing Model

Pull the `main.py` and the `best.pt` files.

1. Connect your camera to your computer via USB.
2. Connect your Arduino and note the port it appears on.
3. Upload your Arduino sketch that receives serial input to your board via Arduino IDE.
4. Place `best.pt` in the same directory as `main.py`.
5. Set your configuration variables in `main.py` as described above.
6. Run the script:

```bash
python main.py
```

A camera window will open. Place an object in the detection zone. The HUD displays the current state, detection streak, and stability counter. Press `q` to quit.

---

## Other Scripts

The repository includes additional scripts for those who wish to retrain or extend the model.

- **`imagecollector.py`** — Captures and saves frames from the camera to build a custom image dataset.
- **`trainmodel.py`** — Trains a new YOLO11 model on a labeled dataset exported from Roboflow.

These scripts are not required to run the sorter with the existing model.
