import sys
import time
from pathlib import Path
import cv2
import numpy as np

from ultralytics import YOLO
from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D
from cv_core.tracking.tracker import MultiTargetTracker
from cv_core.visualizer.bev_radar import BEVRadarRenderer
from cv_core.kinetics.pose_classifier import PoseKineticClassifier

class TripwireInteractiveDemo:
    """
    Tripwire Interactive Computer Vision Showcase GUI.
    Native high-performance desktop interface with on-screen HUD,
    dual-view perspective + BEV radar, live sliders, and interactive controls.
    """
    def __init__(self):
        self.videos = {
            "1": "data/videos/walking_cctv.mp4",
            "2": "data/videos/cctv_sample.mp4",
            "3": "data/videos/solutions_ci_demo.mp4",
            "0": "0"  # Webcam
        }
        self.current_video_key = "1"
        self.paused = False
        self.show_grid = True
        self.show_pose = True
        self.conf_threshold = 0.35
        self.speed_alarm_kmh = 4.0

        # Load models
        print("Initializing YOLOv8 & CV Components...")
        self.detector = YOLO("yolov8n.pt")
        self.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
        self.pose_classifier = PoseKineticClassifier()
        self.radar = BEVRadarRenderer(radar_size=480, world_extent_m=20.0, grid_step_m=2.0)

        # Planar Homography Transformer
        self.transformer = HomographyTransformer()
        calib_file = Path("data/calibration.json")
        if calib_file.exists():
            self.transformer = HomographyTransformer.load_calibration(calib_file)
        else:
            self.transformer.set_homography(np.eye(3))

        # Metric Ground Zones
        self.zones = [
            MetricZone3D("z1", "RESTRICTED VAULT", "polygon", [(3.0, 3.0), (12.0, 3.0), (12.0, 10.0), (3.0, 10.0)]),
            MetricZone3D("tw1", "PERIMETER FENCE", "tripwire", [(0.0, 6.0), (16.0, 6.0)])
        ]

        self.cap = None
        self._load_source(self.current_video_key)

    def _load_source(self, key: str):
        if self.cap is not None:
            self.cap.release()
        self.current_video_key = key
        src = self.videos.get(key, self.videos["1"])
        cap_target = int(src) if src.isdigit() else src
        self.cap = cv2.VideoCapture(cap_target)
        self.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
        print(f"Loaded feed [{key}]: {src}")

    def run(self):
        window_name = "TRIPWIRE CV DEMO — PERSPECTIVE (LEFT) + BEV RADAR (RIGHT)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1440, 720)

        # Interactive trackbars
        def on_conf(val):
            self.conf_threshold = max(0.1, val / 100.0)

        def on_speed(val):
            self.speed_alarm_kmh = float(val)

        cv2.createTrackbar("Confidence (%)", window_name, int(self.conf_threshold * 100), 95, on_conf)
        cv2.createTrackbar("Speed Alarm (km/h)", window_name, int(self.speed_alarm_kmh), 20, on_speed)

        t_fps = time.time()
        fps_counter = 0
        fps = 0.0

        last_frame = None

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # Loop video
                    src = self.videos.get(self.current_video_key, self.videos["1"])
                    if not src.isdigit():
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    if not ret or frame is None:
                        time.sleep(0.03)
                        continue

                last_frame = frame.copy()
                fps_counter += 1
                if (time.time() - t_fps) >= 1.0:
                    fps = fps_counter / (time.time() - t_fps)
                    fps_counter = 0
                    t_fps = time.time()
            else:
                frame = last_frame.copy() if last_frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)

            # 1. Projective Ground Grid Overlay
            if self.show_grid and self.transformer.is_calibrated:
                segments = self.transformer.project_ground_grid_to_image(frame.shape[:2], grid_step_m=2.0)
                grid_overlay = frame.copy()
                for pt1, pt2 in segments:
                    cv2.line(grid_overlay, pt1, pt2, (60, 80, 90), 1, cv2.LINE_AA)
                cv2.addWeighted(grid_overlay, 0.4, frame, 0.6, 0, frame)

            # 2. YOLOv8 Detection
            results = self.detector(frame, classes=[0], conf=self.conf_threshold, verbose=False)
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

            # 3. Kalman Tracker & Homography
            active_tracks = self.tracker.update(detections, homography_transformer=self.transformer)

            # 4. Threat Kinetics & Dual-View Annotation
            radar_tracks = []
            alarm_triggered = False

            for trk in active_tracks:
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
                                    alarm_triggered = True
                                    break
                        elif z.contains_point(w_pos[0], w_pos[1]):
                            is_breach = True
                            alarm_triggered = True
                            break

                speed_kmh = float(trk.get("speed", 0.0))
                if speed_kmh > self.speed_alarm_kmh:
                    is_breach = True
                    alarm_triggered = True

                radar_tracks.append({
                    "track_id": trk["track_id"],
                    "world_pos": w_pos,
                    "world_trail": trk.get("world_trail", []),
                    "speed_kmh": speed_kmh,
                    "vx": trk.get("vx", 0.0),
                    "vy": trk.get("vy", 0.0),
                    "is_breaching": is_breach
                })

                # Perspective Bounding Box
                x1, y1, x2, y2 = map(int, trk["bbox"])
                box_color = (0, 0, 255) if is_breach else (0, 240, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (int(trk["foot_point"][0]), int(trk["foot_point"][1])), 5, (0, 0, 255), -1)

                tag = f"#{trk['track_id']} | {speed_kmh:.1f} km/h"
                cv2.putText(frame, tag, (x1, max(16, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

            # 5. Render BEV Radar
            radar_canvas = self.radar.render(self.zones, radar_tracks)

            # 6. Composite Dual-View
            dual_view = BEVRadarRenderer.create_dual_view(frame, radar_canvas)
            dh, dw = dual_view.shape[:2]

            # 7. Render Telemetry HUD Bar
            cv2.rectangle(dual_view, (0, 0), (dw, 40), (12, 16, 20), -1)
            feed_label = f"FEED: [{self.current_video_key}] {Path(self.videos[self.current_video_key]).name} | FPS: {fps:.1f} | INTRUDERS: {len(active_tracks)}"
            cv2.putText(dual_view, feed_label, (16, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 240, 255), 1, cv2.LINE_AA)

            if alarm_triggered:
                cv2.putText(dual_view, "⚠️ PERIMETER BREACH ACTIVE", (dw // 2 - 120, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            controls_str = "[1/2/3/0] Switch Video | [SPACE] Pause | [G] Grid | [S] Snapshot | [Q] Quit"
            (cw, _), _ = cv2.getTextSize(controls_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(dual_view, controls_str, (dw - cw - 16, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 180, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, dual_view)
            key = cv2.waitKey(20 if not self.paused else 100) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('g'):
                self.show_grid = not self.show_grid
            elif key in [ord('1'), ord('2'), ord('3'), ord('0')]:
                self._load_source(chr(key))
            elif key == ord('s'):
                snap_path = f"data/snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_path, dual_view)
                print(f"✅ Saved snapshot to {snap_path}")

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = TripwireInteractiveDemo()
    demo.run()
