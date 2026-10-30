import sys
import time
import threading
from pathlib import Path

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

app = FastAPI(title="Tripwire CV Live Demo Showcase")

class WebStreamEngine:
    def __init__(self):
        self.videos = {
            "walking": "data/videos/walking_cctv.mp4",
            "hallway": "data/videos/cctv_sample.mp4",
            "crossing": "data/videos/solutions_ci_demo.mp4",
            "webcam": "0"
        }
        self.active_feed = "walking"
        self.conf = 0.35
        self.detector = YOLO("yolov8n.pt")
        self.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
        self.radar = BEVRadarRenderer(radar_size=480, world_extent_m=20.0, grid_step_m=2.0)
        self.transformer = HomographyTransformer()
        calib_file = Path("data/calibration.json")
        if calib_file.exists():
            self.transformer = HomographyTransformer.load_calibration(calib_file)
        else:
            self.transformer.set_homography(np.eye(3))

        self.zones = [
            MetricZone3D("z1", "RESTRICTED VAULT", "polygon", [(3.0, 3.0), (12.0, 3.0), (12.0, 10.0), (3.0, 10.0)]),
            MetricZone3D("tw1", "PERIMETER FENCE", "tripwire", [(0.0, 6.0), (16.0, 6.0)])
        ]
        self.lock = threading.Lock()
        self.cap = None
        self._init_cap()

    def _init_cap(self):
        if self.cap:
            self.cap.release()
        src = self.videos[self.active_feed]
        target = int(src) if src.isdigit() else src
        self.cap = cv2.VideoCapture(target)
        self.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)

    def set_feed(self, feed_name: str):
        with self.lock:
            if feed_name in self.videos:
                self.active_feed = feed_name
                self._init_cap()

    def get_frame(self) -> bytes:
        with self.lock:
            if not self.cap or not self.cap.isOpened():
                self._init_cap()

            if not self.cap or not self.cap.isOpened():
                return b""

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not self.videos[self.active_feed].isdigit():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret or frame is None:
                    return b""

        # Detection & Tracking
        results = self.detector(frame, classes=[0], conf=self.conf, verbose=False)
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "foot_point": ((x1 + x2) / 2.0, float(y2))
                    })

        tracks = self.tracker.update(detections, homography_transformer=self.transformer)
        radar_tracks = []
        for trk in tracks:
            w_pos = trk.get("world_pos")
            is_breach = False
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
            x1, y1, x2, y2 = map(int, trk["bbox"])
            box_color = (0, 0, 255) if is_breach else (0, 240, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"#{trk['track_id']} | {speed_kmh:.1f} km/h", (x1, max(16, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        radar_img = self.radar.render(self.zones, radar_tracks)
        dual_view = BEVRadarRenderer.create_dual_view(frame, radar_img)
        
        # Resize to 720p for fast streaming
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
    select, button { background: #1e293b; color: #f1f5f9; border: 1px solid #334155; padding: 8px 14px; border-radius: 8px; font-size: 14px; cursor: pointer; }
    select:hover, button:hover { background: #334155; }
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
        <option value="walking">Walking Corridor CCTV</option>
        <option value="hallway">Hallway Intrusion</option>
        <option value="crossing">Pedestrian Crossing</option>
        <option value="webcam">Live USB Webcam</option>
      </select>
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
      <div class="card-val">4D Kalman + Hungarian</div>
    </div>
    <div class="card">
      <div class="card-label">Action Classification</div>
      <div class="card-val">17-Joint Keypoint Pose</div>
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
