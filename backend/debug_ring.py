"""For each clip: hue class counts in bands at increasing distance from the lamp core."""
import sys
import cv2
import numpy as np
sys.path.insert(0, "backend")
from config import CLIPS

def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

name = sys.argv[1]
cap = cv2.VideoCapture(f"data/clips/{name}.mp4")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W, Hh = int(cap.get(3)), int(cap.get(4))
mask = np.zeros((Hh, W), np.uint8)
cv2.fillPoly(mask, [np.array(CLIPS[name]["light"], np.int32)], 255)

for idx in (0, n // 3, 2 * n // 3):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    if not ok:
        continue
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    inside = mask > 0
    vf = v.astype(np.float32)
    contrast = vf - cv2.blur(vf, (31, 31))
    core = (inside & (v >= 235) & (contrast >= 45)).astype(np.uint8)
    nc = int(core.sum())
    print(f"{name} f{idx}: core pixels={nc}")
    if nc == 0:
        continue
    prev = cv2.dilate(core, disk(1)) > 0
    for lo, hi in ((1, 3), (3, 5), (5, 8), (8, 12)):
        band = (cv2.dilate(core, disk(hi)) > 0) & (cv2.dilate(core, disk(lo)) == 0)
        band &= inside & (s >= 60) & (v >= 100)
        red = int((band & ((h <= 9) | (h >= 168))).sum())
        yel = int((band & (h >= 10) & (h <= 35)).sum())
        grn = int((band & (h >= 45) & (h <= 100)).sum())
        print(f"   band {lo}-{hi}px:  red={red:4d}  yellow={yel:4d}  green={grn:4d}")