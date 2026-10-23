"""Step 5: classify the traffic-light state (red/yellow/green) in a marked region."""
import sys
from collections import Counter, deque

import cv2
import numpy as np

from config import CLIPS

MIN_PIXELS = 10   # minimum bright colored pixels to call a color
VOTE_FRAMES = 5   # majority vote over this many frames

name = sys.argv[1] if len(sys.argv) > 1 else "clip1"
video = f"data/clips/{name}.mp4"
poly = np.array(CLIPS[name]["light"], dtype=np.int32)

cap = cv2.VideoCapture(video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24
writer = cv2.VideoWriter(
    f"outputs/{name}_light.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

region = np.zeros((height, width), np.uint8)
cv2.fillPoly(region, [poly], 255)

COLORS = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0), "unknown": (200, 200, 200)}


def classify(frame):
    """Return (state, pixel counts) using bright, saturated pixels in the region."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (val > 150) & (sat > 80) & (region > 0)
    counts = {
        "red": int((bright & ((hue < 10) | (hue > 165))).sum()),
        "yellow": int((bright & (hue >= 10) & (hue < 35)).sum()),
        "green": int((bright & (hue >= 35) & (hue < 95)).sum()),
    }
    best = max(counts, key=counts.get)
    return (best if counts[best] >= MIN_PIXELS else "unknown"), counts


recent = deque(maxlen=VOTE_FRAMES)
timeline = []  # (seconds, new_state)
totals = Counter()
last_state = None
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    raw, counts = classify(frame)
    recent.append(raw)
    state = Counter(recent).most_common(1)[0][0]  # majority vote
    totals[state] += 1
    if state != last_state:
        timeline.append((round(frame_idx / fps, 2), state))
        last_state = state

    cv2.polylines(frame, [poly], True, COLORS[state], 3)
    cv2.putText(frame, f"LIGHT: {state.upper()}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLORS[state], 3)
    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()

print(f"Clip: {video} ({frame_idx} frames, {fps:.0f} FPS)")
print("Light state changes (seconds -> state):")
for t, s in timeline:
    print(f"  {t:6.2f}s  {s}")
print("Frames per state:", dict(totals))
print(f"Saved: outputs/{name}_light.mp4")