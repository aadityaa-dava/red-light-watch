"""Step 3: detect vehicles with YOLOv8n and track them with ByteTrack."""
import sys
import time
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO

VEHICLES = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck

video = sys.argv[1] if len(sys.argv) > 1 else "data/clips/traffic.mp4"
max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 1200  # 0 = whole video
out_path = Path("outputs") / f"{Path(video).stem}_tracked.mp4"

model = YOLO("yolov8n.pt")
info = sv.VideoInfo.from_video_path(video)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

writer = cv2.VideoWriter(
    str(out_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    info.fps,
    (info.width, info.height),
)

unique_ids = set()
frames = 0
start = time.time()
for frame in sv.get_video_frames_generator(video):
    # Ultralytics runs YOLO + ByteTrack; persist=True keeps IDs across frames
    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        classes=VEHICLES,
        verbose=False,
    )[0]
    detections = sv.Detections.from_ultralytics(result)

    annotated = frame.copy()
    if detections.tracker_id is not None and len(detections) > 0:
        labels = [f"#{tid}" for tid in detections.tracker_id]
        unique_ids.update(int(t) for t in detections.tracker_id)
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)
    writer.write(annotated)

    frames += 1
    if max_frames and frames >= max_frames:
        break

elapsed = time.time() - start
writer.release()
print(f"Video: {video} ({info.width}x{info.height}, {info.fps:.0f} FPS source)")
print(f"Frames processed: {frames}")
print(f"Unique vehicle IDs: {len(unique_ids)}")
print(f"Processing speed: {frames / elapsed:.1f} FPS")
print(f"Saved: {out_path}")