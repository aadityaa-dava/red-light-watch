# RedLightWatch

Edge-based detection of vehicles running red lights at an intersection.
Runs on an NVIDIA Jetson Orin Nano (8GB) using YOLOv8n, ByteTrack tracking,
and a stop-line crossing rule combined with the traffic-light state.

## Docker image
- Image name: `redlightwatch:dev`
- Base image (JetPack 6): `ultralytics/ultralytics:latest-jetson-jetpack6`

## Build and run (on the Jetson)
```bash
docker build -t redlightwatch:dev .
docker run --runtime nvidia -it --rm redlightwatch:dev
```

## Layout
- `backend/`  Python scripts for detection and decision logic
- `data/clips/`  test videos (not committed)
- `outputs/`  annotated videos, snapshots, CSV reports (not committed)
- `requirements.txt`, `Dockerfile`

## Reference
Roboflow: Count Objects Crossing Lines with Computer Vision
https://roboflow.com/blog/count-objects-crossing-lines
