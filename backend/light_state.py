"""Print the traffic-light state timeline for a clip (uses LightReader) and save an annotated video."""
import sys
import cv2
import numpy as np
sys.path.insert(0, "backend")
from config import CLIPS
from light import LightReader

COLORS = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0), "unknown": (200, 200, 200)}

name = sys.argv[1]
poly = np.array(CLIPS[name]["light"], np.int32)
cap = cv2.VideoCapture(f"data/clips/{name}.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
reader = LightReader(CLIPS[name]["light"], W, H)
out = cv2.VideoWriter(f"outputs/{name}_light.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

prev, i = None, 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = i / fps
    state, _ = reader.update(frame, t)
    if state != prev:
        print(f"{name}  t={t:5.2f}s  frame {i:4d}: {state}")
        prev = state
    color = COLORS[state]
    cv2.polylines(frame, [poly], True, color, 3)
    cv2.putText(frame, f"LIGHT: {state.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    out.write(frame)
    i += 1
out.release()
print(f"Saved: outputs/{name}_light.mp4")