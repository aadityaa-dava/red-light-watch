"""Traffic-light state reader: classifies red/yellow/green inside a marked region."""
from collections import Counter, deque

import cv2
import numpy as np


class LightReader:
    def __init__(self, polygon, width, height, vote_frames=5, min_pixels=10):
        self.poly = np.array(polygon, dtype=np.int32)
        self.region = np.zeros((height, width), np.uint8)
        cv2.fillPoly(self.region, [self.poly], 255)
        self.min_pixels = min_pixels
        self.recent = deque(maxlen=vote_frames)
        self.state = "unknown"
        self.red_since = None  # time when the light turned red

    def _raw_state(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        bright = (val > 150) & (sat > 80) & (self.region > 0)
        counts = {
            "red": int((bright & ((hue < 10) | (hue > 165))).sum()),
            "yellow": int((bright & (hue >= 10) & (hue < 35)).sum()),
            "green": int((bright & (hue >= 35) & (hue < 95)).sum()),
        }
        best = max(counts, key=counts.get)
        return best if counts[best] >= self.min_pixels else "unknown"

    def update(self, frame, t):
        """Return (stable_state, seconds_since_red or None)."""
        self.recent.append(self._raw_state(frame))
        new = Counter(self.recent).most_common(1)[0][0]  # majority vote
        if new != self.state:
            if new == "red":
                # red at the very start of a clip: assume it has been red a while
                self.red_since = -1e9 if t < 1.0 else t
            else:
                self.red_since = None
            self.state = new
        since = (t - self.red_since) if (self.state == "red" and self.red_since is not None) else None
        return self.state, since