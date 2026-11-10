# RedLightWatch

Edge red-light violation detector for an NVIDIA Jetson Orin Nano (8 GB, JetPack 6).
It detects and tracks vehicles with YOLOv8n + ByteTrack, reads the traffic-light colour,
and flags every vehicle that crosses the stop line while the light is red. Output is an
annotated video, an events CSV and a snapshot of each violation.

## How it works

1. **Detect + track**: YOLOv8n (cars, motorcycles, buses, trucks) with ByteTrack, run as a TensorRT FP16 engine.
2. **Read the light**: the user-defined light region is cropped, lamp blobs are found by brightness, and the colour is read from the ring around the lamp core (overexposed lamps look white in the centre). A 5-frame majority vote smooths the result.
3. **Stop-line crossing**: a vehicle's bottom-centre point must move from the approach side to the far side of the line, in the direction of travel. A 10 px band around the line ignores jitter.
4. **Verdict** at the moment of crossing: green = LEGAL, yellow or red for under 1 s = LATE_ENTRY, red for longer = VIOLATION (NEEDS_REVIEW if detection confidence is under 0.5 or the light is unreadable).
5. A vehicle that reappears behind the line within 1.5 s is downgraded to NEEDS_REVIEW. Intersections with 3 or more violations in a clip are flagged HIGH-RISK.

## Project layout

```
backend/
  config.py            stop line and light region for each clip
  light.py             traffic-light colour reader
  light_state.py       light-only timeline + video (for checking the reader)
  detect_violations.py main pipeline (detect, track, judge, save)
  evaluate.py          scores outputs against ground truth
  profile_pipeline.py  per-stage timing
data/clips/            test videos clip1..clip9 (not in git)
outputs/               videos, events CSVs, violation snapshots (not in git)
requirements.txt
```

## Setup (Jetson, Docker)

Docker image: `ultralytics/ultralytics:latest-jetson-jetpack6`

```bash
git clone https://github.com/aadityaa-dava/red-light-watch.git
cd red-light-watch
docker run -it --rm --runtime nvidia \
  -v $(pwd):/app -w /app \
  -p 8889:8889 -p 8501:8501 \
  ultralytics/ultralytics:latest-jetson-jetpack6
# inside the container:
pip install -r requirements.txt
```

Optional TensorRT engine (about 10 minutes to build, much faster inference):

```bash
yolo export model=yolov8n.pt format=engine half=True imgsz=960 device=0
mv yolov8n.engine yolov8n_960.engine
```

## Run

```bash
python3 backend/detect_violations.py clip1                     # PyTorch model
RLW_IMGSZ=960 RLW_MODEL=yolov8n_960.engine \
  python3 backend/detect_violations.py clip1                   # TensorRT engine
RLW_NOVIDEO=1 ... python3 backend/detect_violations.py clip1   # skip the output video (faster)
python3 backend/evaluate.py                                    # score all clips
```

Outputs: `outputs/<clip>_violations.mp4`, `outputs/<clip>_events.csv`,
`outputs/violations/<clip>_id<N>_<EVENT>.jpg`.

## Results

Per-clip counts against hand-labelled ground truth (TensorRT FP16, 960 px input):

| Metric | Value |
|---|---|
| Clips | 9 |
| True positives / false positives / false negatives | 7 / 0 / 0 |
| Precision / recall | 100% / 100% |
| Speed, annotated video written | 8 to 12 FPS |
| Speed, video off (clip1) | 13.1 FPS |

The 15 FPS target was not reached; speed is still real-time for the 1080p source clips on the Orin Nano.

## Limitations

- Thresholds and line/light regions were tuned on these same 9 clips; there is no held-out test set, so real-world accuracy will be lower.
- Evaluation compares counts per clip, not individual cars.
- The stop line and light region are set by hand for each camera.
- Clips 7 to 9 have variable frame rate and camera zoom or pan, which can move the line relative to the road.
- Night, glare and heavy rain make light reading and detection less reliable.

## Frontend

A Streamlit dashboard runs the detector on a sample clip or on an uploaded video. For an upload you mark the stop line and the traffic light on a preview frame, then press Run detection. It shows the violation counters, the annotated video, the events table, the violation snapshots and a CSV download.

```bash
apt-get update && apt-get install -y ffmpeg      # needed to play the annotated video in the browser
pip install streamlit streamlit-image-coordinates
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

Open `http://<jetson-ip>:8501` in a browser (`192.168.55.1` over the USB link).

## Results files

`results/` holds the final run on all 9 clips: `evaluation.csv`, one events CSV per clip, the violation snapshots and the run log (`run_all_final.log`).
