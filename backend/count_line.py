"""Step 4: count tracked vehicles crossing a line (the future stop line)."""
import csv
import sys
import time
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO

VEHICLES = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck

# Line endpoints in pixels (x, y). Starting values come from the Roboflow
# article's example for this video. Adjust after checking outputs/line_preview.jpg
LINE_START = (10, 341)
LINE_END = (1263, 349)

video = sys.argv[1] if len(sys.argv) > 1 else "data/clips/traffic.mp4"
max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 1200  # 0 = whole video
out_path = Path("outputs") / f"{Path(video).stem}_line.mp4"
csv_path = Path("outputs") / f"{Path(video).stem}_crossings.csv"

model = YOLO("yolov8n.pt")
info = sv.VideoInfo.from_video_path(video)
line = sv.LineZone(
    start=sv.Point(*LINE_START),
    end=sv.Point(*LINE_END),
    triggering_anchors=[sv.Position.BOTTOM_CENTER],  # where the car meets the road
)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
line_annotator = sv.LineZoneAnnotator(thickness=2)

writer = cv2.VideoWriter(
    str(out_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    info.fps,
    (info.width, info.height),
)

events = []  # (frame, seconds, track_id, direction)
frames = 0
start = time.time()
for frame_idx, frame in enumerate(sv.get_video_frames_generator(video)):
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
        crossed_in, crossed_out = line.trigger(detections)
        for i, tid in enumerate(detections.tracker_id):
            if crossed_in[i] or crossed_out[i]:
                direction = "in" if crossed_in[i] else "out"
                events.append((frame_idx, round(frame_idx / info.fps, 2), int(tid), direction))
        labels = [f"#{tid}" for tid in detections.tracker_id]
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)
    annotated = line_annotator.annotate(annotated, line_counter=line)

    if frame_idx == 0:
        cv2.imwrite("outputs/line_preview.jpg", annotated)
    writer.write(annotated)

    frames += 1
    if max_frames and frames >= max_frames:
        break

elapsed = time.time() - start
writer.release()

with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "seconds", "track_id", "direction"])
    w.writerows(events)

print(f"Video: {video} ({info.width}x{info.height}, {info.fps:.0f} FPS source)")
print(f"Frames processed: {frames}")
print(f"Crossed IN: {line.in_count}  |  Crossed OUT: {line.out_count}  |  Total: {line.in_count + line.out_count}")
print(f"Crossing events logged: {len(events)} -> {csv_path}")
print(f"Processing speed: {frames / elapsed:.1f} FPS")
print(f"Saved: {out_path}")