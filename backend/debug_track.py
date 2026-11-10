"""Print one car's track around the stop line and WHY a crossing is or is not counted.

usage: python3 backend/debug_track.py clip1 8      (clip name, track id)
d_* = distance to the stop line in px (+ behind the line, - beyond it)
seg = where the car is along the marked line (0 = start, 1 = end; counted only if -0.05..1.05)
fwd/cos = what the direction check sees over the last 8 frames (needs fwd>=4 and cos>=0.5)
"""
import sys
from collections import deque
import numpy as np
import supervision as sv
from ultralytics import YOLO
sys.path.insert(0, "backend")
from config import CLIPS

name, want = sys.argv[1], int(sys.argv[2])
cfg = CLIPS[name]
video = f"data/clips/{name}.mp4"
info = sv.VideoInfo.from_video_path(video)
fps = info.fps
A = np.array(cfg["line"][0], float)
B = np.array(cfg["line"][1], float)
D = B - A
L2 = float(D @ D)
L = np.sqrt(L2)
ref = np.array(cfg.get("approach_point", (info.width / 2, info.height - 1)), float)
cref = D[0] * (ref[1] - A[1]) - D[1] * (ref[0] - A[0])
SIGN = 1 if cref > 0 else -1
NORMAL = np.array([D[1], -D[0]]) / L * SIGN


def dist(p):
    return SIGN * (D[0] * (p[1] - A[1]) - D[1] * (p[0] - A[0])) / L


model = YOLO("yolov8n.pt")
hist = deque(maxlen=8)
print(f"{'t(s)':>6} {'frame':>5} {'cx':>5} {'d_top':>6} {'d_ctr':>6} {'d_bot':>6} {'seg':>6} {'fwd':>6} {'cos':>5} {'conf':>5}")
for idx, frame in enumerate(sv.get_video_frames_generator(video)):
    r = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.35, imgsz=1280,
                    classes=[2, 3, 5, 7], agnostic_nms=True, verbose=False)[0]
    det = sv.Detections.from_ultralytics(r)
    if det.tracker_id is None:
        continue
    for i, tid in enumerate(det.tracker_id):
        if int(tid) != want:
            continue
        x1, y1, x2, y2 = det.xyxy[i]
        cx = (x1 + x2) / 2
        p = np.array([cx, y2])                 # bottom-center, the point the pipeline tracks
        hist.append(p)
        seg = float(((p - A) @ D) / L2)
        m = hist[-1] - hist[0]
        fwd = float(m @ NORMAL)
        mag = float(np.hypot(*m))
        cs = fwd / mag if mag > 0 else 0.0
        print(f"{idx / fps:6.2f} {idx:5d} {cx:5.0f} {dist((cx, y1)):6.0f} {dist((cx, (y1 + y2) / 2)):6.0f} "
              f"{dist(p):6.0f} {seg:6.2f} {fwd:6.1f} {cs:5.2f} {det.confidence[i]:5.2f}")