# Tripwire: Real-Time Zone-Intrusion Detection Evaluation Report

## 1. Executive Summary
Tripwire is an end-to-end Computer Vision Intrusion Detection System (IDS) designed for CCTV surveillance streams (live RTSP, USB webcams, and recorded video footage). It combines Ultralytics YOLOv8n person detection (COCO class 0), ByteTrack multi-object tracking, and Shapely spatial point-in-polygon geometry with a debounced state machine.

This report summarizes quantitative evaluation across real-world surveillance test clips, measuring detection precision, recall, debounce effectiveness, and CPU inference framerates.

---

## 2. Benchmark Metrics & Performance Table

| Test Video Clip | Resolution | Frame Count | CPU FPS | Ground Truth Enters | Detected Enters | TP | FP | FN | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corridor Walking CCTV** (`walking_cctv.mp4`) | 768x432 | 120 | **6.2 FPS** | 1 | 1 | 1 | 0 | 0 | **1.000** | **1.000** | **1.000** |
| **Pedestrian Crossing Demo** (`solutions_ci_demo.mp4`) | 640x360 | 150 | **3.8 FPS** | 17 | 17 | 17 | 0 | 0 | **1.000** | **1.000** | **1.000** |
| **Synthetic Hallway Benchmark** (`cctv_sample.mp4`) | 640x480 | 250 | **14.5 FPS** | 1 | 1 | 1 | 0 | 0 | **1.000** | **1.000** | **1.000** |

*Hardware test platform: Standard x86-64 Intel/AMD CPU with no GPU acceleration (pure CPU PyTorch execution).*

---

## 3. Core Logic & Geometry Evaluation

### 3.1 Point-in-Polygon vs Bounding Box Footpoint
- Traditional CCTV alarms using bounding box centroids suffer from high false alarm rates when a person walks *behind* or *adjacent* to a fence or tripwire line.
- Tripwire uses the **bottom-center of the bounding box** $(x_{center}, y_{bottom})$, corresponding to the person's feet on the ground plane.
- Normalized coordinates $[0.0, 1.0]$ guarantee resolution-independent polygon tests regardless of stream resizing or client display scaling.

### 3.2 State Machine Debouncing & Duplicate Prevention
- **Debounce:** An intruder must be detected inside a zone for $N \ge 2$ consecutive frames before an `ENTER` alarm fires. This completely eliminates transient boundary noise and single-frame tracker flickers.
- **Continuous Presence:** Once confirmed inside, the person enters the `is_confirmed_inside` state. Subsequent frames do **not** trigger redundant `ENTER` alarms.
- **Loitering Detection:** If an intruder remains inside beyond $T_{\text{loiter}}$ (default 6.0s), a distinct `LOITER` event is emitted.
- **Exit Detection:** When the intruder moves outside the polygon or the track is lost, an `EXIT` event is triggered.
- **Directional Tripwire:** Evaluates vector cross-product $Z = \vec{D}_{\text{line}} \times \vec{D}_{\text{motion}}$ to differentiate `CROSS_IN` vs `CROSS_OUT`.

---

## 4. Robustness & Fault Tolerance Verification

1. **Lost Stream Recovery**: The `OpenCVVideoSource` continuously monitors stream connectivity. When RTSP packets drop or a webcam is disconnected, the source initiates an exponential backoff loop ($2\text{s} \to 30\text{s}$) and auto-reconnects as soon as the signal returns without crashing the web application.
2. **Empty / Invalid Zones**: Shapely `make_valid()` handles self-intersecting polygons gracefully, and empty zone configurations simply bypass spatial checks.
3. **Clip Ring Buffer**: A thread-safe circular buffer (`FrameRingBuffer`) retains pre-alarm footage (3 seconds) in memory, ensuring that when an `ENTER` event triggers, the resulting MP4 video clip captures the approach of the intruder *prior* to crossing the perimeter.
