"""List every candidate lamp blob (size, position) and the ring hue counts around it."""
import sys
import cv2
import numpy as np
sys.path.insert(0, "backend")
from config import CLIPS
from light import LightReader, _disk

name = sys.argv[1]
cap = cv2.VideoCapture(f"data/clips/{name}.mp4")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W, H = int(cap.get(3)), int(cap.get(4))
mask = np.zeros((H, W), np.uint8)
cv2.fillPoly(mask, [np.array(CLIPS[name]["light"], np.int32)], 255)

for idx in (0, n // 4, n // 2, 3 * n // 4):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    if not ok:
        continue
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    inside = mask > 0
    vf = v.astype(np.float32)
    contrast = vf - cv2.blur(vf, (31, 31))
    cand = (inside & (v >= 235) & (contrast >= 45)).astype(np.uint8)
    cnt, labels, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    print(f"{name} frame {idx}: {cnt - 1} blobs")
    rows = []
    for i in range(1, cnt):
        x, y, w, hh, area = stats[i]
        if area < 2:
            continue
        b = (labels == i).astype(np.uint8)
        ring = (cv2.dilate(b, _disk(5)) > 0) & (cv2.dilate(b, _disk(1)) == 0)
        ring &= inside & (s >= 60) & (v >= 100)
        red = int((ring & ((h <= 9) | (h >= 168))).sum())
        yel = int((ring & (h >= 10) & (h <= 35)).sum())
        grn = int((ring & (h >= 45) & (h <= 100)).sum())
        rows.append((area, int(cents[i][0]), int(cents[i][1]), w, hh, red, yel, grn))
    for r in sorted(rows, reverse=True)[:8]:
        print(f"   area={r[0]:5d} at ({r[1]:4d},{r[2]:4d}) size {r[3]}x{r[4]}  ring red={r[5]:3d} yellow={r[6]:3d} green={r[7]:3d}")