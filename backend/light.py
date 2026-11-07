"""Traffic-light color reader: find lamp blobs (round first), read hue from a thin ring around each."""
import cv2
import numpy as np
from collections import deque, Counter


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


class LightReader:
    def __init__(self, polygon, width, height, vote_frames=5, min_pixels=10):
        self.mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(self.mask, [np.array(polygon, np.int32)], 255)
        self.min_pixels = min_pixels
        self.votes = deque(maxlen=vote_frames)
        self.state = "unknown"
        self.red_since = None
        self.seen_non_red = False

    def _lamp_blobs(self, v, inside, contrast):
        cand = (inside & (v >= 235) & (contrast >= 45)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)

        def pick(min_a, max_a, lo_ar, hi_ar, min_fill):
            keep = np.zeros_like(cand)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if not (min_a <= area <= max_a):
                    continue
                if not (lo_ar <= w / max(h, 1) <= hi_ar):
                    continue
                if area / float(w * h) < min_fill:
                    continue
                keep[labels == i] = 1
            return keep

        keep = pick(20, 1500, 0.6, 1.7, 0.6)     # round, solid lamps (daytime)
        if keep.sum() == 0:
            keep = pick(2, 400, 0.3, 3.5, 0.3)   # small / odd-shaped lamps
        if keep.sum() == 0:
            keep = pick(2, 4000, 0.3, 3.5, 0.3)  # big glare blobs (night)
        return keep

    def raw_state(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        inside = self.mask > 0
        vf = v.astype(np.float32)
        contrast = vf - cv2.blur(vf, (31, 31))
        core = self._lamp_blobs(v, inside, contrast)
        if core.sum() == 0:
            return "unknown"
        ring = (cv2.dilate(core, _disk(5)) > 0) & (cv2.dilate(core, _disk(1)) == 0)
        ring &= inside & (s >= 60) & (v >= 100)
        red = ring & ((h <= 9) | (h >= 168))
        yellow = ring & (h >= 10) & (h <= 35)
        green = ring & (h >= 45) & (h <= 100)
        score = {"red": int(red.sum()), "yellow": int(yellow.sum()), "green": int(green.sum())}
        best = max(score, key=score.get)
        return best if score[best] >= self.min_pixels else "unknown"

    def update(self, frame, t):
        self.votes.append(self.raw_state(frame))
        new = Counter(self.votes).most_common(1)[0][0]
        # real signals never go red -> yellow (only green -> yellow -> red): treat as noise
        if new == "yellow" and self.state == "red":
            new = "red"
        self.state = new
        if self.state in ("yellow", "green"):
            self.seen_non_red = True
        if self.state == "red":
            if self.red_since is None:
                # a clip that opens on red has been red for an unknown time: assume long
                self.red_since = t if self.seen_non_red else t - 999.0
            return self.state, t - self.red_since
        self.red_since = None
        return self.state, 0.0