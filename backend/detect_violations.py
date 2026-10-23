"""Step 6: red-light violation detection (tracking + stop line + light state)."""
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import CLIPS
from light import LightReader

VEHICLES = [2, 3, 5, 7]       # COCO: car, motorcycle, bus, truck
CONF = 0.35                   # ignore detections below this
CONF_CONFIRM = 0.5            # below this, a crossing goes to NEEDS_REVIEW
IMGSZ = 1280                  # larger input helps with small night-time cars
GRACE_SECONDS = 1.0           # crossing this soon after red is a late entry
HIGH_RISK_THRESHOLD = 3       # violations per clip that flag the intersection

COLORS = {  # BGR
    "VIOLATION": (0, 0, 255),
    "LATE_ENTRY": (0, 215, 255),
    "NEEDS_REVIEW": (0, 140, 255),
    "LEGAL": (0, 200, 0),
    None: (200, 200, 200),
}

name = sys.argv[1] if len(sys.argv) > 1 else "clip1"
cfg = CLIPS[name]
video = f"data/clips/{name}.mp4"
Path("outputs/violations").mkdir(parents=True, exist_ok=True)

info = sv.VideoInfo.from_video_path(video)
fps = info.fps

# --- stop line geometry -----------------------------------------------------
A = np.array(cfg["line"][0], dtype=float)
B = np.array(cfg["line"][1], dtype=float)
D = B - A
LEN2 = float(D @ D)
ref = np.array(cfg.get("approach_point", (info.width / 2, info.height - 1)), dtype=float)


def side_of(p):
    cross = D[0] * (p[1] - A[1]) - D[1] * (p[0] - A[0])
    return 1 if cross > 0 else (-1 if cross < 0 else 0)


APPROACH_SIDE = side_of(ref)  # cars start on this side and cross to the other


def within_segment(p):
    t = ((p - A) @ D) / LEN2
    return -0.05 <= t <= 1.05


def judge(state, since_red, conf):
    if state == "green":
        return "LEGAL"
    if state == "unknown":
        return "NEEDS_REVIEW"
    if state == "yellow":
        return "LATE_ENTRY"
    since = since_red if since_red is not None else 1e9
    if since <= GRACE_SECONDS:
        return "LATE_ENTRY"
    return "VIOLATION" if conf >= CONF_CONFIRM else "NEEDS_REVIEW"


# --- main loop --------------------------------------------------------------
model = YOLO("yolov8n.pt")
light = LightReader(cfg["light"], info.width, info.height)
writer = cv2.VideoWriter(
    f"outputs/{name}_violations.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (info.width, info.height)
)
line_pts = (tuple(int(v) for v in A), tuple(int(v) for v in B))
light_poly = np.array(cfg["light"], dtype=np.int32)

prev_side, done, status, events = {}, set(), {}, []
counts = Counter()
flash_text, flash_until = "", 0.0
start = time.time()
frames = 0

for idx, frame in enumerate(sv.get_video_frames_generator(video)):
    t = idx / fps
    state, since_red = light.update(frame, t)
    result = model.track(
        frame, persist=True, tracker="bytetrack.yaml",
        conf=CONF, imgsz=IMGSZ, classes=VEHICLES, verbose=False,
    )[0]
    det = sv.Detections.from_ultralytics(result)

    new_events, boxes = [], []
    if det.tracker_id is not None and len(det) > 0:
        anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        for i, tid in enumerate(det.tracker_id):
            tid = int(tid)
            p = anchors[i]
            s = side_of(p)
            prev = prev_side.get(tid)
            if s != 0:
                prev_side[tid] = s
            if tid not in done and prev == APPROACH_SIDE and s == -APPROACH_SIDE and within_segment(p):
                done.add(tid)
                conf = float(det.confidence[i])
                kind = judge(state, since_red, conf)
                status[tid] = kind
                counts[kind] += 1
                ev = {"time_s": round(t, 2), "track_id": tid, "event": kind,
                      "light": state, "confidence": round(conf, 2), "snapshot": ""}
                new_events.append(ev)
                events.append(ev)
                flash_text, flash_until = f"{kind}  #{tid}", t + 2.0
            boxes.append((det.xyxy[i].astype(int), tid))

    # --- draw ---
    out = frame.copy()
    cv2.line(out, line_pts[0], line_pts[1], (255, 255, 0), 3)
    cv2.polylines(out, [light_poly], True, COLORS.get({"red": "VIOLATION", "yellow": "LATE_ENTRY", "green": "LEGAL"}.get(state)), 3)
    for (x1, y1, x2, y2), tid in boxes:
        color = COLORS[status.get(tid)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"#{tid}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(out, f"LIGHT: {state.upper()}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(out, f"VIOLATIONS: {counts['VIOLATION']}  LATE: {counts['LATE_ENTRY']}  REVIEW: {counts['NEEDS_REVIEW']}",
                (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    if t < flash_until:
        cv2.putText(out, flash_text, (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)

    for ev in new_events:
        if ev["event"] != "LEGAL":
            path = f"outputs/violations/{name}_id{ev['track_id']}_{ev['event']}.jpg"
            cv2.imwrite(path, out)
            ev["snapshot"] = path
    writer.write(out)
    frames += 1

writer.release()
elapsed = time.time() - start

csv_path = f"outputs/{name}_events.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["time_s", "track_id", "event", "light", "confidence", "snapshot"])
    w.writeheader()
    w.writerows(events)

print(f"Clip: {video} ({frames} frames, {fps:.0f} FPS source)")
print("Events (first crossing of each car):")
for ev in events:
    print(f"  {ev['time_s']:6.2f}s  car #{ev['track_id']:<4} {ev['event']:<13} light={ev['light']:<8} conf={ev['confidence']}")
print(f"Totals: {dict(counts)}")
if counts["VIOLATION"] >= HIGH_RISK_THRESHOLD:
    print("INTERSECTION FLAGGED: HIGH-RISK")
print(f"Processing speed: {frames / elapsed:.1f} FPS")
print(f"Saved: outputs/{name}_violations.mp4, {csv_path}")