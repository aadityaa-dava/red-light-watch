"""Print info for every clip and save a preview frame from each."""
from pathlib import Path

import cv2

clips = sorted(Path("data/clips").glob("*.mp4"))
if not clips:
    print("No clips found in data/clips/")

for clip in clips:
    cap = cv2.VideoCapture(str(clip))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = total / fps if fps else 0
    print(f"{clip.name}: {width}x{height}, {fps:.0f} FPS, {total} frames, {seconds:.0f} s")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps))  # frame at about 1 second
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(f"outputs/preview_{clip.stem}.jpg", frame)
    cap.release()