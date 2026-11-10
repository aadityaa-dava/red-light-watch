"""Time each stage of the pipeline per frame, to see what limits FPS.

usage: RLW_MODEL=yolov8n.engine python3 backend/profile_pipeline.py clip1
"""
import os
import sys
import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

sys.path.insert(0, "backend")
from config import CLIPS
from light import LightReader

name = sys.argv[1] if len(sys.argv) > 1 else "clip1"
N = 60
cfg = CLIPS[name]
video = f"data/clips/{name}.mp4"
info = sv.VideoInfo.from_video_path(video)
model = YOLO(os.environ.get("RLW_MODEL", "yolov8n.pt"), task="detect")
light = LightReader(cfg["light"], info.width, info.height)
writer = cv2.VideoWriter("outputs/_profile.mp4", cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height))
poly = np.array(cfg["light"], dtype=np.int32)

# warm-up so one-time setup is not counted
frame0 = next(iter(sv.get_video_frames_generator(video)))
model.track(frame0, persist=False, imgsz=1280, classes=[2, 3, 5, 7], verbose=False)

names = ["decode", "light", "detect+track", "draw", "write"]
tot = dict.fromkeys(names, 0.0)
gen = iter(sv.get_video_frames_generator(video))
for idx in range(N):
    t0 = time.perf_counter()
    frame = next(gen)
    t1 = time.perf_counter()
    light.update(frame, idx / info.fps)
    t2 = time.perf_counter()
    r = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.35, imgsz=1280,
                    classes=[2, 3, 5, 7], agnostic_nms=True, verbose=False)[0]
    det = sv.Detections.from_ultralytics(r)
    t3 = time.perf_counter()
    out = frame.copy()
    cv2.polylines(out, [poly], True, (0, 0, 255), 3)
    for x1, y1, x2, y2 in det.xyxy.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(out, "LIGHT", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    t4 = time.perf_counter()
    writer.write(out)
    t5 = time.perf_counter()
    for k, d in zip(names, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
        tot[k] += d
writer.release()
total = sum(tot.values())
print(f"\nAverage per frame over {N} frames ({info.width}x{info.height}):")
for k in names:
    print(f"  {k:<14}{tot[k] / N * 1000:7.1f} ms   {tot[k] / total:4.0%}")
print(f"  {'TOTAL':<14}{total / N * 1000:7.1f} ms   ->  {N / total:.1f} FPS")