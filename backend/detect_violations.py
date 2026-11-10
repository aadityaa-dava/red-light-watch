"""Step 6: red-light violation detection (tracking + stop line + light state)."""
import csv
import sys
import time
from collections import Counter, deque
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

HIST_FRAMES = 8               # frames of motion used to check driving direction
MIN_FORWARD = 4.0             # px a car must move toward the far side of the line
MIN_COS = 0.2                 # motion must point within ~78 deg of the approach direction (roads run diagonally in the image)
REVERSE_SECONDS = 1.5         # a car that returns behind the line within this time is downgraded
DEDUP_SECONDS = 2.0           # a 2nd crossing this soon, overlapping an earlier car, is the same vehicle
DEDUP_IOU = 0.5               # box overlap above which two tracks count as one vehicle

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

# which point of each car's box is tracked against the stop line (bottom = rear of a car driving away)
ANCHORS = {"bottom": sv.Position.BOTTOM_CENTER, "center": sv.Position.CENTER, "top": sv.Position.TOP_CENTER}
ANCHOR = sys.argv[2] if len(sys.argv) > 2 else cfg.get("anchor", "bottom")

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

# unit vector perpendicular to the line, pointing from the approach side to the far side
NORMAL = np.array([D[1], -D[0]]) / np.sqrt(LEN2) * APPROACH_SIDE


def within_segment(p):
    t = ((p - A) @ D) / LEN2
    return -0.05 <= t <= 1.05


def moving_forward(history):
    """True if the car has clearly been driving from the approach side toward the far side."""
    if len(history) < 2:
        return False
    m = history[-1] - history[0]
    fwd = float(m @ NORMAL)
    mag = float(np.hypot(m[0], m[1]))
    return mag > 0 and fwd >= MIN_FORWARD and fwd / mag >= MIN_COS


def iou(a, b):
    """Intersection-over-union of two xyxy boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


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
hist, pending, last_box = {}, {}, {}
counts = Counter()
flash_text, flash_until = "", 0.0
start = time.time()
frames = 0


def record(tid, kind, state, conf, t, new_events):
    """Record a crossing immediately, so the label, counter and snapshot appear at once."""
    global flash_text, flash_until
    status[tid] = kind
    counts[kind] += 1
    ev = {"time_s": round(t, 2), "track_id": tid, "event": kind, "light": state,
          "confidence": round(conf, 2), "note": "", "snapshot": ""}
    new_events.append(ev)
    events.append(ev)
    flash_text, flash_until = f"{kind}  #{tid}", t + 2.0
    return ev


def reverse(tid, ev):
    """The car came back behind the line: downgrade the verdict to NEEDS_REVIEW."""
    global flash_text
    counts[ev["event"]] -= 1
    counts["NEEDS_REVIEW"] += 1
    old_snap = ev["snapshot"]
    ev["event"], ev["note"] = "NEEDS_REVIEW", "returned behind line"
    status[tid] = "NEEDS_REVIEW"
    flash_text = f"NEEDS_REVIEW  #{tid}"
    if old_snap and Path(old_snap).exists():
        new_snap = f"outputs/violations/{name}_id{tid}_NEEDS_REVIEW.jpg"
        Path(old_snap).rename(new_snap)
        ev["snapshot"] = new_snap


for idx, frame in enumerate(sv.get_video_frames_generator(video)):
    t = idx / fps
    state, since_red = light.update(frame, t)
    result = model.track(
        frame, persist=True, tracker="bytetrack.yaml",
        conf=CONF, imgsz=IMGSZ, classes=VEHICLES, agnostic_nms=True, verbose=False,
    )[0]
    det = sv.Detections.from_ultralytics(result)

    new_events, boxes, cur_side = [], [], {}
    if det.tracker_id is not None and len(det) > 0:
        anchors = det.get_anchors_coordinates(ANCHORS[ANCHOR])
        for i, tid in enumerate(det.tracker_id):
            tid = int(tid)
            p = np.array(anchors[i], dtype=float)
            hist.setdefault(tid, deque(maxlen=HIST_FRAMES)).append(p)
            s = side_of(p)
            cur_side[tid] = s
            last_box[tid] = det.xyxy[i]
            prev = prev_side.get(tid)
            if s != 0:
                crossed = (tid not in done and prev == APPROACH_SIDE
                           and s == -APPROACH_SIDE and within_segment(p))
                if crossed and not moving_forward(hist[tid]):
                    pass  # sideways, backwards or jitter: not a real crossing, keep waiting
                else:
                    prev_side[tid] = s
                    if crossed:
                        done.add(tid)
                        # same vehicle under a second track id? (recent event + heavily overlapping box)
                        duplicate = any(
                            e["track_id"] != tid and t - e["time_s"] <= DEDUP_SECONDS
                            and e["track_id"] in last_box
                            and iou(det.xyxy[i], last_box[e["track_id"]]) > DEDUP_IOU
                            for e in events if e["event"] != "LEGAL"
                        )
                        if not duplicate:
                            conf = float(det.confidence[i])
                            kind = judge(state, since_red, conf)
                            ev = record(tid, kind, state, conf, t, new_events)
                            if kind != "LEGAL":
                                pending[tid] = {"t0": t, "ev": ev}
            boxes.append((det.xyxy[i].astype(int), tid))

    # --- a crossing that is undone within REVERSE_SECONDS becomes NEEDS_REVIEW ---
    for tid in list(pending):
        pd = pending[tid]
        age = t - pd["t0"]
        if cur_side.get(tid) == APPROACH_SIDE:
            reverse(tid, pd["ev"])
            del pending[tid]
        elif age >= REVERSE_SECONDS:
            del pending[tid]          # stayed past the line: verdict stands

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

events.sort(key=lambda e: e["time_s"])
csv_path = f"outputs/{name}_events.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["time_s", "track_id", "event", "light", "confidence", "note", "snapshot"])
    w.writeheader()
    w.writerows(events)

print(f"Clip: {video} ({frames} frames, {fps:.0f} FPS source)")
print("Events (first crossing of each car):")
for ev in events:
    note = f"  [{ev['note']}]" if ev["note"] else ""
    print(f"  {ev['time_s']:6.2f}s  car #{ev['track_id']:<4} {ev['event']:<13} light={ev['light']:<8} conf={ev['confidence']}{note}")
print(f"Totals: { {k: v for k, v in counts.items() if v} }")
if counts["VIOLATION"] >= HIGH_RISK_THRESHOLD:
    print("INTERSECTION FLAGGED: HIGH-RISK")
print(f"Processing speed: {frames / elapsed:.1f} FPS")
print(f"Saved: outputs/{name}_violations.mp4, {csv_path}")