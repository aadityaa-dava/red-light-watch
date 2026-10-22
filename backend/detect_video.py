"""Step 2: run YOLOv8n vehicle detection on a video and save an annotated copy."""
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

VEHICLES = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck

video = sys.argv[1] if len(sys.argv) > 1 else "data/clips/traffic.mp4"
out_path = Path("outputs") / f"{Path(video).stem}_detected.mp4"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
writer = cv2.VideoWriter(
    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height)
)

frames = 0
detections = 0
start = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = model(frame, conf=0.35, classes=VEHICLES, verbose=False)[0]
    detections += len(result.boxes)
    writer.write(result.plot())
    frames += 1

elapsed = time.time() - start
cap.release()
writer.release()
print(f"Video: {video} ({width}x{height}, {video_fps:.0f} FPS source)")
print(f"Frames processed: {frames}")
print(f"Total vehicle detections: {detections}")
print(f"Processing speed: {frames / elapsed:.1f} FPS")
print(f"Saved: {out_path}")
