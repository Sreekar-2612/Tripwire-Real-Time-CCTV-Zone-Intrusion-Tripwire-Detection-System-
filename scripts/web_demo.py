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
import torch
from fastapi import FastAPI, Response, Request, HTTPException
import os
import base64
import hmac
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
from contextlib import asynccontextmanager

from ultralytics import YOLO
from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D, get_default_zones
from cv_core.tracking.tracker import MultiTargetTracker
from cv_core.visualizer.bev_radar import BEVRadarRenderer
from cv_core.kinetics.pose_classifier import PoseKineticClassifier, draw_pose_skeleton
from cv_core.video import FrameClock
from cv_core.monitoring import ZoneMonitor

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    yield
    if engine.cap:
        try:
            engine.cap.release()
        except Exception:
            pass

app = FastAPI(title="Tripwire CV Live Demo Showcase", lifespan=lifespan)


@app.middleware('http')
async def protect_camera(request: Request, call_next):
    token = os.environ.get('TRIPWIRE_TOKEN')
    if token:
        try:
            scheme, encoded = request.headers.get('authorization', '').split(' ', 1)
            user, password = base64.b64decode(encoded, validate=True).decode().split(':', 1)
            authorized = scheme.lower() == 'basic' and user == 'operator' and hmac.compare_digest(password.encode(), token.encode())
        except (ValueError, UnicodeError):
            authorized = False
        if not authorized:
            return Response(status_code=401, headers={'WWW-Authenticate': 'Basic realm="Tripwire"'})
    elif request.client is None or request.client.host not in ('127.0.0.1', '::1', 'testclient'):
        return Response('Remote camera access requires TRIPWIRE_TOKEN', status_code=403)
    return await call_next(request)


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
        self.use_pose = False

        # Load models
        pose_model_path = Path("yolov8n-pose.pt")
        self.detector_pose = YOLO(str(pose_model_path)) if pose_model_path.exists() else None
        self.use_pose = self.detector_pose is not None
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

        self.zones = get_default_zones(self.active_feed)
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
        radar_img = self.radar.render(self.zones if self.transformer.is_calibrated else [], [])
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

        self.clock = FrameClock(self.cap, live=str(src).isdigit() or '://' in str(src)) if self.cap is not None else None
        size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) if self.cap is not None else (0, 0)
        self.transformer = HomographyTransformer.for_source(src, size)
        if not self.transformer.is_calibrated and Path("data/calibration.json").exists():
            self.transformer = HomographyTransformer.load_calibration("data/calibration.json")
        self.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=.3)
        self.zones = get_default_zones(self.active_feed)
        self.monitor = ZoneMonitor()
        self.latest_jpeg = None
        self.last_frame_time = float('-inf')

    def set_feed(self, feed_name: str):
        with self.lock:
            if feed_name not in self.videos:
                raise ValueError("Unknown video feed")
            if feed_name in self.videos:
                self.active_feed = feed_name
                self._init_cap()

    def toggle_pose(self, enabled: Optional[bool] = None) -> bool:
        with self.lock:
            if enabled is not None:
                self.use_pose = enabled and self.detector_pose is not None
            else:
                self.use_pose = not self.use_pose and self.detector_pose is not None
            self.latest_jpeg = None
            return self.use_pose

    def get_frame(self) -> bytes:
        # Fix #12: hold the lock for the ENTIRE read + infer + track cycle so
        # two concurrent subscribers cannot race on the same capture and tracker.
        # This serialises all callers; the first request fully processes one frame
        # before the next caller enters, which keeps tracker state consistent.
        with self.lock:
            now = time.time()
            if self.latest_jpeg is not None and time.monotonic() - self.last_frame_time < self.clock.interval:
                return self.latest_jpeg
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
                        self.tracker = MultiTargetTracker(min_hits=2)
                        self.monitor = ZoneMonitor()
                        self.clock = FrameClock(self.cap)
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

            elapsed_dt = self.clock.tick()

            # Model Inference — inside the lock so feed switches cannot insert
            # a stale-feed detection into the newly-reset tracker (fix #12).
            is_pose_active = self.use_pose and (self.detector_pose is not None)
            active_detector = self.detector_pose if is_pose_active else self.detector_standard

            with torch.inference_mode():
                results = active_detector(frame, conf=0.15, imgsz=480, verbose=False)
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
                pose_classifier=self.pose_classifier if is_pose_active else None,
                zones=self.zones, elapsed_dt=elapsed_dt
            )
            alarm_states, _ = self.monitor.update(tracks, self.zones, elapsed_dt)
            display_frame = frame.copy()
            if self.transformer.is_calibrated:
                poly_overlay = display_frame.copy()
                has_polys = False
                for z in self.zones:
                    if z.zone_type == "polygon":
                        try:
                            pts = [self.transformer.world_to_image(x, y) for x, y in z.world_coords]
                            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(poly_overlay, [pts_np], (0, 140, 255))
                            has_polys = True
                        except Exception:
                            pass
                if has_polys:
                    cv2.addWeighted(poly_overlay, 0.30, display_frame, 0.70, 0, display_frame)

                for z in self.zones:
                    if z.zone_type == "polygon":
                        try:
                            pts = [self.transformer.world_to_image(x, y) for x, y in z.world_coords]
                            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(display_frame, [pts_np], True, (0, 185, 255), 2, cv2.LINE_AA)
                            lbl_x = int(min(p[0] for p in pts)) + 8
                            lbl_y = int(min(p[1] for p in pts)) + 20
                            cv2.putText(display_frame, z.name, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
                        except Exception:
                            pass
                    elif z.zone_type == "tripwire":
                        try:
                            p1 = self.transformer.world_to_image(*z.world_coords[0])
                            p2 = self.transformer.world_to_image(*z.world_coords[1])
                            ip1 = (int(p1[0]), int(p1[1]))
                            ip2 = (int(p2[0]), int(p2[1]))
                            cv2.line(display_frame, ip1, ip2, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(display_frame, z.name, (ip1[0] + 10, ip1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
                        except Exception:
                            pass

            radar_tracks = []
            for trk in tracks:
                w_pos = trk.get("world_pos")
                is_breach = alarm_states[trk['track_id']]['is_breaching']
                posture = trk.get('posture', 'unknown')
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
                    draw_pose_skeleton(display_frame, trk["keypoints"])

                x1, y1, x2, y2 = map(int, trk["bbox"])
                box_color = (0, 0, 255) if is_breach else (0, 240, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

                prefix = "🚨 " if is_breach else ""
                label_text = f"{prefix}#{trk['track_id']} | {posture.upper()} | " + (f"{speed_kmh:.1f} km/h" if w_pos is not None else "speed unavailable")
                cv2.putText(display_frame, label_text, (x1, max(16, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2, cv2.LINE_AA)

            radar_img = self.radar.render(self.zones if self.transformer.is_calibrated else [], radar_tracks,
                                          title='GROUND RADAR' if self.transformer.is_calibrated else 'UNCALIBRATED - METRIC ALARMS DISABLED')
            dual_view = BEVRadarRenderer.create_dual_view(display_frame, radar_img)

            # Resize for responsive web streaming
            dual_view_resized = cv2.resize(dual_view, (1200, 500))
            _, jpeg = cv2.imencode(".jpg", dual_view_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.latest_jpeg = jpeg.tobytes()
            self.last_frame_time = time.monotonic()
            return self.latest_jpeg

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
      <div class="card-val" id="poseCard">Pose available when weights are installed</div>
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
    function showPose(d) {
      const btn = document.getElementById('poseBtn');
      btn.innerText = d.available === false ? 'Pose unavailable' : 'Pose Kinetics: ' + (d.use_pose ? 'ON' : 'OFF');
      btn.disabled = d.available === false;
      btn.className = d.use_pose ? 'btn-pose-active' : '';
      document.getElementById('poseCard').innerText = d.use_pose ? '17-Joint Pose AI' : 'YOLOv8-Nano';
    }
    function togglePose() {
      fetch('/api/toggle_pose', { method: 'POST' }).then(r => r.json()).then(showPose);
    }
    fetch('/api/toggle_pose').then(r => r.json()).then(showPose);
  </script>
</body>
</html>
"""

@app.get("/")
def get_dashboard():
    return HTMLResponse(HTML_PAGE)

@app.post("/api/switch_feed")
def switch_feed(feed: str):
    try:
        engine.set_feed(feed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "feed": engine.active_feed}

@app.post("/api/toggle_pose")
def toggle_pose():
    state = engine.toggle_pose()
    return {"use_pose": state, "available": engine.detector_pose is not None}

@app.get("/api/toggle_pose")
def get_toggle_pose():
    return {"use_pose": engine.use_pose, "available": engine.detector_pose is not None}

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
    # Fix #11: default to loopback (127.0.0.1) so the demo is not reachable
    # from other hosts on the network without an explicit operator choice.
    import os
    host = os.environ.get("TRIPWIRE_HOST", "127.0.0.1")
    print(f"🚀 Starting Tripwire Live Web Demo at http://{host}:8000")
    if host not in ('127.0.0.1', 'localhost', '::1') and not os.environ.get('TRIPWIRE_TOKEN'):
        raise RuntimeError('Set TRIPWIRE_TOKEN before exposing camera feeds to the network')
    uvicorn.run(app, host=host, port=8000, proxy_headers=False)
