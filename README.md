# TRIP-WIRE: CCTV tracking, zones, and ground radar

TRIP-WIRE is a Python computer-vision prototype for person detection, multi-target tracking, pose rules, and polygon/tripwire monitoring. It has a Streamlit dashboard, a FastAPI MJPEG demo, and an OpenCV CLI/desktop interface.

The system uses Ultralytics YOLO, a constant-velocity Kalman filter, threshold-aware Hungarian matching with DIoU costs, and a calibrated planar homography. Pose actions are heuristic rules, not a trained action-recognition model.

## Setup

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

Model weights are downloaded on first use. The Dockerfile preloads both `yolov8n.pt` and `yolov8n-pose.pt`. Keep one OpenCV distribution in your environment: desktop preview needs a GUI-enabled build, while `--no-preview` works with headless OpenCV.

## Camera calibration

A ground mapping belongs to one camera and frame size. Without a matching calibration, person tracking continues, but metric speed, ground radar positions, and spatial alarms are disabled. The old unbound `data/calibration.json` is not automatically applied to arbitrary feeds.

```bash
python scripts/calibrate_homography.py --source data/videos/walking_cctv.mp4 --output data/videos/walking_cctv.mp4.calibration.json --width 3 --length 6
```

Click the four corners of a measured floor rectangle in TL, TR, BR, BL order. The tool writes source identity and frame dimensions with the matrix. A `<source>.calibration.json` sidecar loads automatically. A bound `data/calibration.json` loads only for its matching source and size. Recalibrate when the camera moves or its image geometry changes.

Streamlit also accepts a calibration JSON selected explicitly for the current feed. Legacy files without source metadata remain usable only when selected explicitly; the operator must verify their camera and measurement scale. Do not relabel someone else's calibration as belonging to your camera.

## Run

```bash
python scripts/run_cv_pipeline.py --source data/videos/walking_cctv.mp4 --output result.mp4 --no-preview
python scripts/run_cv_pipeline.py --source 0 --pose
python scripts/demo_gui.py
python scripts/web_demo.py
```

The CLI accepts `--calibration FILE` for an explicit calibration and `--pose` for action recognition. The desktop demo uses pose weights when installed. Files use source timestamps/FPS; live cameras use monotonic acquisition time. Lost-track speed uses the complete interval between observations.

Tripwires retain each track's last off-wire point, so touching and returning does not count as crossing. Polygon dwell and tripwires are checked independently. Streamlit records ENTER, LOITER, EXIT, crossings, and action/speed transitions; sustained incidents do not produce a new record every frame. Evidence and logs are bounded session memory, not durable storage.

## Network access

FastAPI listens on loopback by default. Remote requests are denied unless `TRIPWIRE_TOKEN` is configured. For remote deployment set `TRIPWIRE_HOST` deliberately and configure that secret in the process environment. The browser uses HTTP Basic authentication with username `operator` and that token as the password. Protect remote traffic with HTTPS using your deployment proxy; the demo does not terminate TLS. Do not expose the Streamlit service publicly without your platform's authentication boundary.

FastAPI serializes camera processing and shares a short-lived encoded frame between viewers. Processing is demand-driven rather than an always-running capture service. GET `/api/toggle_pose` reads state; POST toggles it. Missing pose weights are reported as unavailable.

## Tests and measurements

```bash
python -m pytest
python scripts/benchmark_models.py --imgsz 640 --runs 50
python evaluation/benchmark_feedback_loop.py
```

`benchmark_models.py` is a **synthetic CPU microbenchmark**, not a YOLO throughput measurement. The feedback script includes real YOLO inference plus separate synthetic component experiments; a sum of separate stages is not measured application throughput.

There are no verified 40.1 FPS, 96.8% MOTA, or 94.7% IDF1 guarantees. Labeled annotations and a reproducible evaluation run are required before publishing detection/identity accuracy. See [evaluation notes](evaluation/EVALUATION_REPORT.md), [original audit](audit/AUDIT_REPORT.md), and [fix verification](audit/VERIFICATION_REPORT.md) for evidence and limits.

## Source layout

- `cv_core/geometry`: homography validation and metric zones.
- `cv_core/tracking`: Kalman state, association, track confirmation, speed, and pose smoothing.
- `cv_core/monitoring.py`: per-camera zone history and incident transitions.
- `cv_core/video.py`: source/live frame timing.
- `cv_core/kinetics`: COCO-keypoint pose rules and drawing.
- `cv_core/visualizer`: ground radar and dual-view composition.
- `scripts`: calibration, CLI, desktop, web, and benchmark entry points.
- `tests`: core and application regression checks.

License: MIT. Calibrate and validate on your own cameras before using alarms operationally.
