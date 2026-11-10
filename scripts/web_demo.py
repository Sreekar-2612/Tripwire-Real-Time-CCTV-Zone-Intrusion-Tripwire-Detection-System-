import sys
import time
import threading
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

from ultralytics import YOLO
from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D
from cv_core.tracking.tracker import MultiTargetTracker
from cv_core.visualizer.bev_radar import BEVRadarRenderer
from cv_core.kinetics.pose_classifier import PoseKineticClassifier, draw_pose_skeleton

app = FastAPI(title="Tripwire CV Live Demo Showcase")

class WebStreamEngine:
    def __init__(self):
        self.videos = {
            "template_tripwire": "data/videos/corridor_tripwire_breach.mp4",
            "template_speed": "data/videos/high_speed_fleeing_breach.mp4",
            "walking": "data/videos/walking_cctv.mp4",
            "hallway": "data/videos/cctv_sample.mp4",
            "crossing": "data/videos/solutions_ci_demo.mp4",
            "webcam": "0"
        }
        self.active_feed = "template_tripwire"
        self.conf = 0.35
        self.use_pose = True

        # Load models
        pose_model_path = Path("yolov8n-pose.pt")
        self.detector_pose = YOLO(str(pose_model_path)) if pose_model_path.exists() else None
        self.detector_standard = YOLO("yolov8n.pt")
        self.pose_classifier = PoseKineticClassifier(keypoint_conf_threshold=0.35)

        self.tracker = MultiTargetTracker(
            min_hits=2,
            max_age=15,
            iou_threshold=0.3,
            high_conf_threshold=0.50,
            low_conf_threshold=0.15,
            ema_alpha=0.35
        )
        self.radar = BEVRadarRenderer(radar_size=480, world_extent_m=20.0, grid_step_m=2.0)
        self.transformer = HomographyTransformer()
        calib_file = Path("data/calibration.json")
        if calib_file.exists():
            self.transformer = HomographyTransformer.load_calibration(calib_file)
        else:
            self.transformer.set_homography(np.eye(3))

        self.zones = [
            MetricZone3D("z1", "RESTRICTED VAULT", "polygon", [(3.5, 1.2), (7.5, 1.2), (7.5, 2.8), (3.5, 2.8)]),
            MetricZone3D("tw1", "PERIMETER FENCE", "tripwire", [(2.0, 2.2), (8.0, 2.2)])
        ]
        self.lock = threading.Lock()
        self.cap = None
        self.last_reconnect_time = 0.0
        self._init_cap()

    def _generate_disconnected_frame(self, reason: str = "CAMERA SIGNAL DISCONNECTED") -> bytes:
        """Generates a fallback status frame when camera feed is unavailable or disconnected."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (20, 20, 25)
        cv2.putText(frame, "WARNING: " + reason, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Feed: [{self.active_feed}] | Auto-reconnecting...", (30, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        radar_img = self.radar.render(self.zones, [])
        dual_view = BEVRadarRenderer.create_dual_view(frame, radar_img)
        ret_enc, buffer = cv2.imencode(".jpg", dual_view)
        return buffer.tobytes() if ret_enc else b""

    def _init_cap(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        src = self.videos.get(self.active_feed, "0")
        try:
            if str(src).isdigit():
                target = int(src)
                self.cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(target)
            else:
                self.cap = cv2.VideoCapture(src)
        except Exception:
            self.cap = None

        self.tracker = MultiTargetTracker(
            min_hits=2,
            max_age=15,
            iou_threshold=0.3,
            high_conf_threshold=0.50,
            low_conf_threshold=0.15,
            ema_alpha=0.35
        )

    def set_feed(self, feed_name: str):
        with self.lock:
            if feed_name in self.videos:
                self.active_feed = feed_name
                self._init_cap()

    def toggle_pose(self, enabled: Optional[bool] = None) -> bool:
        with self.lock:
            if enabled is not None:
                self.use_pose = enabled
            else:
                self.use_pose = not self.use_pose
            return self.use_pose

    def get_frame(self) -> bytes:
        with self.lock:
            now = time.time()
            if not self.cap or not self.cap.isOpened():
                if (now - self.last_reconnect_time) > 2.0:
                    self.last_reconnect_time = now
                    self._init_cap()

            if not self.cap or not self.cap.isOpened():
                return self._generate_disconnected_frame("CAMERA NOT AVAILABLE")

            try:
                ret, frame = self.cap.read()
            except Exception:
                ret, frame = False, None

            if not ret or frame is None:
                src_val = str(self.videos.get(self.active_feed, ""))
                if not src_val.isdigit():
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    except Exception:
                        ret, frame = False, None

                if not ret or frame is None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    self.last_reconnect_time = now
                    return self._generate_disconnected_frame("CAMERA FEED DISCONNECTED")

        # Model Inference
        is_pose_active = self.use_pose and (self.detector_pose is not None)
        active_detector = self.detector_pose if is_pose_active else self.detector_standard

        # Down to 0.15 for ByteTrack Two-Stage Association
        results = active_detector(frame, conf=0.15, verbose=False)
        detections = []
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes
                has_kpts = is_pose_active and getattr(r, "keypoints", None) is not None and len(r.keypoints.data) > 0
                for idx_b, box in enumerate(boxes):
                    cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                    if cls_id != 0:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    kpt = None
                    if has_kpts and idx_b < len(r.keypoints.data):
                        kpt = r.keypoints.data[idx_b].cpu().numpy()
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "foot_point": ((x1 + x2) / 2.0, float(y2)),
                        "keypoints": kpt
                    })

        tracks = self.tracker.update(
            detections,
            homography_transformer=self.transformer,
            pose_classifier=self.pose_classifier if is_pose_active else None
        )
        radar_tracks = []
        for trk in tracks:
            w_pos = trk.get("world_pos")
            is_breach = False
            posture = trk.get("posture", "walking")
            threat_level = trk.get("threat_level", 1)

            # Spatial Zone Breaches
            if w_pos:
                for z in self.zones:
                    if z.zone_type == "tripwire":
                        world_trail = trk.get("world_trail", [])
                        if len(world_trail) >= 2:
                            crossing = z.check_tripwire_crossing(world_trail[-2], world_trail[-1])
                            if crossing is not None:
                                is_breach = True
                                break
                    elif z.contains_point(w_pos[0], w_pos[1]):
                        is_breach = True
                        break

            # Threat / Kinetic Escalation
            if threat_level >= 7 or posture in ("climbing", "crawling", "fallen"):
                is_breach = True

            speed_kmh = float(trk.get("speed", 0.0))
            radar_tracks.append({
                "track_id": trk["track_id"],
                "world_pos": w_pos,
                "world_trail": trk.get("world_trail", []),
                "speed_kmh": speed_kmh,
                "vx": trk.get("vx", 0.0),
                "vy": trk.get("vy", 0.0),
                "is_breaching": is_breach
            })

            # Render Skeleton Keypoints if available
            if is_pose_active and trk.get("keypoints") is not None:
                draw_pose_skeleton(frame, trk["keypoints"])

            x1, y1, x2, y2 = map(int, trk["bbox"])
            box_color = (0, 0, 255) if is_breach else (0, 240, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            label_text = f"#{trk['track_id']} | {posture.upper()} | {speed_kmh:.1f} km/h"
            cv2.putText(frame, label_text, (x1, max(16, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2)

        radar_img = self.radar.render(self.zones, radar_tracks)
        dual_view = BEVRadarRenderer.create_dual_view(frame, radar_img)
        
        # Resize for responsive web streaming
        dual_view_resized = cv2.resize(dual_view, (1200, 500))
        _, jpeg = cv2.imencode(".jpg", dual_view_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

engine = WebStreamEngine()

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Tripwire — Live Computer Vision & BEV Radar Demo</title>
  <style>
    body { margin: 0; background-color: #0b0f17; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: column; align-items: center; padding: 20px; }
    .header { width: 100%; max-width: 1200px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 12px; margin-bottom: 20px; }
    .title { font-size: 22px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
    .badge { background: #0284c7; color: #fff; font-size: 11px; padding: 3px 8px; border-radius: 999px; text-transform: uppercase; }
    .controls { display: flex; gap: 12px; align-items: center; }
    select, button { background: #1e293b; color: #f1f5f9; border: 1px solid #334155; padding: 8px 14px; border-radius: 8px; font-size: 14px; cursor: pointer; transition: all 0.2s ease; }
    select:hover, button:hover { background: #334155; }
    .btn-pose-active { background: #0284c7; border-color: #38bdf8; font-weight: 600; }
    .stream-container { width: 100%; max-width: 1200px; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.7); border: 1px solid #1e293b; background: #000; }
    img { width: 100%; display: block; }
    .cards { width: 100%; max-width: 1200px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 20px; }
    .card { background: #111827; border: 1px solid #1f2937; padding: 16px; border-radius: 10px; }
    .card-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px; }
    .card-val { font-size: 20px; font-weight: 700; color: #38bdf8; }
  </style>
</head>
<body>
  <div class="header">
    <div class="title">
      <span>🛡️ TRIPWIRE CV SURVEILLANCE</span>
      <span class="badge">Live BEV Radar Demo</span>
    </div>
    <div class="controls">
      <label for="feed">Feed:</label>
      <select id="feed" onchange="changeFeed(this.value)">
        <option value="template_tripwire">Template: Corridor Tripwire Crossing Breach</option>
        <option value="template_speed">Template: High-Speed Fleeing Breach</option>
        <option value="walking">Walking Corridor CCTV (Full 60s)</option>
        <option value="hallway">Hallway Intrusion</option>
        <option value="crossing">Pedestrian Crossing</option>
        <option value="webcam">Live USB / Laptop Webcam</option>
      </select>
      <button id="poseBtn" class="btn-pose-active" onclick="togglePose()">⚡ Pose Kinetics: ON</button>
    </div>
  </div>

  <div class="stream-container">
    <img src="/stream" alt="Tripwire Live Stream">
  </div>

  <div class="cards">
    <div class="card">
      <div class="card-label">Perspective Geometry</div>
      <div class="card-val">DLT Homography</div>
    </div>
    <div class="card">
      <div class="card-label">Tracking Engine</div>
      <div class="card-val">ByteTrack + EMA</div>
    </div>
    <div class="card">
      <div class="card-label">Action Classification</div>
      <div class="card-val" id="poseCard">17-Joint Pose AI</div>
    </div>
    <div class="card">
      <div class="card-label">Ground Grid Extent</div>
      <div class="card-val">20m x 20m Radar</div>
    </div>
  </div>

  <script>
    function changeFeed(val) {
      fetch('/api/switch_feed?feed=' + val, { method: 'POST' });
    }
    function togglePose() {
      fetch('/api/toggle_pose', { method: 'POST' })
        .then(r => r.json())
        .then(d => {
          const btn = document.getElementById('poseBtn');
          const card = document.getElementById('poseCard');
          if (d.use_pose) {
            btn.innerText = '⚡ Pose Kinetics: ON';
            btn.className = 'btn-pose-active';
            card.innerText = '17-Joint Pose AI';
          } else {
            btn.innerText = '⚡ Pose Kinetics: OFF';
            btn.className = '';
            card.innerText = 'YOLOv8-Nano';
          }
        });
    }
  </script>
</body>
</html>
"""

@app.get("/")
def get_dashboard():
    return HTMLResponse(HTML_PAGE)

@app.post("/api/switch_feed")
def switch_feed(feed: str):
    engine.set_feed(feed)
    return {"status": "ok", "feed": feed}

@app.post("/api/toggle_pose")
def toggle_pose():
    state = engine.toggle_pose()
    return {"use_pose": state}

@app.get("/api/toggle_pose")
def get_toggle_pose():
    state = engine.toggle_pose()
    return {"use_pose": state}

@app.get("/stream")
def video_stream():
    def mjpeg_gen():
        while True:
            frame_bytes = engine.get_frame()
            if frame_bytes:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.03)

    return StreamingResponse(
        mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    print("🚀 Starting Tripwire Live Web Demo at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
