"""Print hue/sat/brightness of the highest-contrast pixels inside a clip's light polygon."""
import sys
import cv2
import numpy as np
sys.path.insert(0, "backend")
from config import CLIPS

name = sys.argv[1]
cap = cv2.VideoCapture(f"data/clips/{name}.mp4")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
h_, w_ = int(cap.get(4)), int(cap.get(3))
mask = np.zeros((h_, w_), np.uint8)
cv2.fillPoly(mask, [np.array(CLIPS[name]["light"], np.int32)], 255)

for idx in (0, n // 3, 2 * n // 3):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    if not ok:
        continue
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    inside = mask > 0
    v = hsv[..., 2].astype(np.float32)
    local = v - cv2.blur(v, (25, 25))
    score = np.where(inside, local, -999)
    top = np.argsort(score.ravel())[-40:]
    ys, xs = np.unravel_index(top, score.shape)
    H, S, V = hsv[ys, xs, 0], hsv[ys, xs, 1], hsv[ys, xs, 2]
    print(f"{name} frame {idx}: H med={int(np.median(H))} [{H.min()}-{H.max()}]  "
          f"S med={int(np.median(S))}  V med={int(np.median(V))}  "
          f"at x~{int(np.median(xs))} y~{int(np.median(ys))}")