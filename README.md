# RedLightWatch

Edge-based detection of vehicles running red lights at an intersection.
Runs on an NVIDIA Jetson Orin Nano (8GB) with YOLOv8n and ByteTrack. A
violation is a tracked vehicle that crosses the stop line while the traffic
light is stably red.

## How it works

1. **Detect** vehicles (car, motorcycle, bus, truck) with YOLOv8n.
2. **Track** each vehicle with ByteTrack so every car keeps one ID.
3. **Read the traffic light** by checking the red/yellow/green pixels inside a
   marked region, with a 5-frame majority vote to avoid flicker.
4. **Check the stop line** using the bottom-center of each vehicle's box and
   the direction of travel.
5. **Decide**, once per vehicle ID:

| Situation at the moment of crossing | Result |
|-------------------------------------|--------|
| Light red for more than 1 second, confident detection | `VIOLATION` |
| Light yellow, or red for 1 second or less | `LATE_ENTRY` (not a violation) |
| Light unknown, or low-confidence detection | `NEEDS_REVIEW` |
| Light green | `LEGAL` (not flagged) |

If a clip has 3 or more violations, the intersection is flagged `HIGH-RISK`.

Handling uncertainty: detections below 0.35 confidence are ignored; crossings
with confidence below 0.5 go to `NEEDS_REVIEW`; the light state is
`unknown` if not enough colored pixels are seen, and no violation can be
issued in that state.

## Docker image

- **Image name:** `redlightwatch:dev`
- **Base image (JetPack 6):** `ultralytics/ultralytics:latest-jetson-jetpack6`
- For JetPack 5, build with
  `--build-arg BASE_IMAGE=ultralytics/ultralytics:latest-jetson-jetpack5`

PyTorch and OpenCV come from the base image (GPU builds), so they are not in
`requirements.txt`.

## Setup and run (on the Jetson)

```bash
git clone https://github.com/aadityaa-dava/red-light-watch.git
cd red-light-watch

# build the image and check the environment
docker build -t redlightwatch:dev .
docker run --runtime nvidia -it --rm redlightwatch:dev

# run the violation detector on a clip (clips go in data/clips/)
docker run --runtime nvidia -it --rm --ipc=host \
  -v "$(pwd)":/app redlightwatch:dev \
  python3 backend/detect_violations.py clip1
```

Outputs are written to `outputs/`:
- `<clip>_violations.mp4`: annotated video with line, light state, and counters
- `<clip>_events.csv`: one row per crossing (time, car ID, event, light, confidence)
- `violations/`: snapshot images of each flagged crossing

## Project layout

```
backend/
  check_env.py          environment check (GPU, YOLO, supervision)
  inspect_clips.py      print clip info and save a preview frame
  detect_video.py       Step 2: vehicle detection
  track_video.py        Step 3: detection + ByteTrack tracking
  count_line.py         Step 4: line-crossing counts (in/out)
  light_state.py        Step 5: traffic-light color in a marked region
  light.py              LightReader class used by the violation script
  detect_violations.py  Step 6: the red-light violation rule
  config.py             per-clip stop line and light region (pixels)
frontend/               Streamlit app (added after the backend submission)
data/clips/             test videos (not committed)
outputs/                results (not committed)
Dockerfile
requirements.txt
```

## Adding a clip

1. Put `clipN.mp4` in `data/clips/`.
2. Run `python3 backend/inspect_clips.py` to save `outputs/preview_clipN.jpg`.
3. Open the preview in [polygonzone.roboflow.com](https://polygonzone.roboflow.com).
   Press **L** to mark the stop line (two points) and **P** to mark the
   traffic-light region (four points).
4. Add the coordinates to `backend/config.py`:
```python
   "clipN": {
       "line": [(x1, y1), (x2, y2)],
       "light": [(x, y), (x, y), (x, y), (x, y)],
   },
```
   Vehicles are assumed to approach from the bottom of the frame. If they
   come from elsewhere, add `"approach_point": (x, y)` to the clip's entry,
   using a point on the approach side of the line.
5. Run `python3 backend/detect_violations.py clipN`.

## Data

Test videos are not stored in this repository.

- **Source:** TODO: add the YouTube URL and title of the compilation video
  (the traffic-camera footage was trimmed into separate clips).
- **Clips:** `clip1.mp4` to `clip5.mp4`, 1920x1080, 24 FPS, 11 to 13 seconds
  each. TODO: add the start and end timestamp of each clip in the source video.
- **Sample video** for detection, tracking, and line counting:
  `https://media.roboflow.com/traffic.mp4` (1280x720, 60 FPS).

## Results

| Clip | Violations reported | Notes |
|------|---------------------|-------|
| clip1 | 1 (car #8 at 7.42 s, light red, conf 0.91) | |
| clip2 | 2 (car #27 at 5.63 s, conf 0.55; car #33 at 5.76 s, conf 0.87) | only one real car: the tracker gave the same truck two IDs, so one report is a false alarm |
| clip3 | 1 (car #48 at 6.13 s, conf 0.90) | |
| clip4 | 1 (car #59 at 4.63 s, conf 0.92) | |
| clip5 | 1 (car #7 at 7.72 s, conf 0.89) | |

Speed on the Jetson Orin Nano (PyTorch, input size 1280): 4.6 to 4.8 FPS on
every clip.

TODO: add TensorRT speed and precision/recall after the evaluation.

## Limitations

- The stop line and light region are marked by hand for each camera view.
- The light reader uses color thresholds tuned on night footage; daytime clips
  may need threshold tuning.
- Small, distant vehicles at night are hard to detect; a larger input size
  helps but lowers the speed.
- The tracker can give one vehicle two IDs (seen in `clip2`), which produces a
  duplicate report for a single car.

## References

- Roboflow: [Count Objects Crossing Lines with Computer Vision](https://roboflow.com/blog/count-objects-crossing-lines)
- Ultralytics YOLOv8 and ByteTrack tracking
- Roboflow `supervision` library
