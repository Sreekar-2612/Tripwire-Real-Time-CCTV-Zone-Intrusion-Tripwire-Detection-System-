import sys
import time
import threading
from pathlib import Path

try:
    import winsound
except ImportError:
    winsound = None

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from ultralytics import YOLO
from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D, get_default_zones
from cv_core.tracking.tracker import MultiTargetTracker
from cv_core.visualizer.bev_radar import BEVRadarRenderer
from cv_core.kinetics.pose_classifier import PoseKineticClassifier, draw_pose_skeleton
from cv_core.video import FrameClock
from cv_core.monitoring import ZoneMonitor
from cv_core.audio.siren import SirenAlertController

class TripwireInteractiveDemo:
    """
    Tripwire Interactive Computer Vision Showcase GUI.
    Native high-performance desktop interface with on-screen HUD,
    dual-view perspective + BEV radar, live sliders, and interactive controls.
    """
    def __init__(self):
        self.videos = {
            "1": "data/videos/corridor_tripwire_breach.mp4",
            "2": "data/videos/high_speed_fleeing_breach.mp4",
            "3": "data/videos/walking_cctv.mp4",
            "4": "data/videos/cctv_sample.mp4",
            "5": "data/videos/solutions_ci_demo.mp4",
            "6": "data/videos/flagship_multi_threat_benchmark.mp4",
            "0": "0"  # Webcam
        }
        self.current_video_key = "1"
        self.paused = False
        self.show_grid = True
        self.show_pose = Path("yolov8n-pose.pt").exists()
        self.conf_threshold = 0.35
        self.speed_alarm_kmh = 4.0
        self.audio_alarm_enabled = True
        self.last_alarm_sound_time = 0.0
        self.siren_controller = SirenAlertController(enabled=self.audio_alarm_enabled)

        # Load models
        print("Initializing YOLOv8 & CV Components...")
        self.detector = YOLO("yolov8n-pose.pt" if self.show_pose else "yolov8n.pt")
        self.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
        self.pose_classifier = PoseKineticClassifier()
        self.radar = BEVRadarRenderer(radar_size=480, world_extent_m=20.0, grid_step_m=2.0)

        # Planar Homography Transformer
        self.transformer = HomographyTransformer()
        # Metric Ground Zones
        self.zones = get_default_zones(self.videos[self.current_video_key])

        self.cap = None
        self._load_source(self.current_video_key)

    def _load_source(self, key: str):
        if self.cap is not None:
            self.cap.release()
        self.current_video_key = key
        src = self.videos.get(key, self.videos["1"])
        self.zones = get_default_zones(src)
        cap_target = int(src) if src.isdigit() else src
        self.cap = cv2.VideoCapture(cap_target)
        self.clock = FrameClock(self.cap, live=src.isdigit() or '://' in src)
        size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.transformer = HomographyTransformer.for_source(src, size)
        if not self.transformer.is_calibrated and Path("data/calibration.json").exists():
            self.transformer = HomographyTransformer.load_calibration("data/calibration.json")
        self.monitor = ZoneMonitor()
        self.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
        print(f"Loaded feed [{key}]: {src}")

    def _trigger_audio_alarm(self):
        """
        Dispatches non-blocking audible siren beeps in a daemon thread upon threat detection.
        Throttled to prevent overlapping audio loops or CV thread blocking.
        """
        if hasattr(self, "siren_controller") and self.siren_controller is not None:
            self.siren_controller.set_enabled(getattr(self, "audio_alarm_enabled", True))
            self.siren_controller.sync_with_events(breach_count=1)
            self.siren_controller.trigger_non_blocking_beep()
            return

        if not getattr(self, "audio_alarm_enabled", True):
            return
        now = time.time()
        if now - getattr(self, "last_alarm_sound_time", 0.0) < 0.65:
            return
        self.last_alarm_sound_time = now

        def _beep_worker():
            if winsound is not None:
                try:
                    winsound.Beep(900, 150)
                    winsound.Beep(650, 150)
                except Exception:
                    pass
            else:
                try:
                    print('\a', end='', flush=True)
                except Exception:
                    pass

        threading.Thread(target=_beep_worker, daemon=True).start()

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

        try:
            last_view = np.zeros((480, 1280, 3), dtype=np.uint8)
            while True:
                if self.paused:
                    cv2.imshow(window_name, last_view)
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord(' '):
                        self.paused = False
                        self.clock = FrameClock(self.cap, live=self.videos[self.current_video_key].isdigit())
                    continue
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        # Loop video
                        src = self.videos.get(self.current_video_key, self.videos["1"])
                        if not src.isdigit():
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.clock = FrameClock(self.cap)
                            self.tracker = MultiTargetTracker(min_hits=2)
                            self.monitor = ZoneMonitor()
                            ret, frame = self.cap.read()
                        if not ret or frame is None:
                            # Fix #19: process key events even on read failure so Q/feed-switch work
                            status_frame = np.zeros((480, 1280, 3), dtype=np.uint8)
                            cv2.putText(status_frame, "CAMERA DISCONNECTED — waiting for frame...",
                                        (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 200), 2, cv2.LINE_AA)
                            cv2.imshow(window_name, status_frame)
                            key = cv2.waitKey(100) & 0xFF
                            if key == ord('q'):
                                break
                            elif chr(key) in self.videos:
                                self._load_source(chr(key))
                            continue

                    last_frame = frame.copy()
                    fps_counter += 1
                    if (time.time() - t_fps) >= 1.0:
                        fps = fps_counter / (time.time() - t_fps)
                        fps_counter = 0
                        t_fps = time.time()
                else:
                    frame = last_frame.copy() if last_frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)

                elapsed_dt = self.clock.tick()
                clean_frame = frame.copy()

                # 1. Projective Ground Grid & Zone Overlay
                if self.show_grid and self.transformer.is_calibrated:
                    segments = self.transformer.project_ground_grid_to_image(frame.shape[:2], grid_step_m=2.0)
                    grid_overlay = frame.copy()
                    for pt1, pt2 in segments:
                        cv2.line(grid_overlay, pt1, pt2, (60, 80, 90), 1, cv2.LINE_AA)
                    cv2.addWeighted(grid_overlay, 0.4, frame, 0.6, 0, frame)

                if self.transformer.is_calibrated:
                    poly_overlay = frame.copy()
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
                        cv2.addWeighted(poly_overlay, 0.30, frame, 0.70, 0, frame)

                    for z in self.zones:
                        if z.zone_type == "polygon":
                            try:
                                pts = [self.transformer.world_to_image(x, y) for x, y in z.world_coords]
                                pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.polylines(frame, [pts_np], True, (0, 185, 255), 2, cv2.LINE_AA)
                                lbl_x = int(min(p[0] for p in pts)) + 8
                                lbl_y = int(min(p[1] for p in pts)) + 20
                                cv2.putText(frame, z.name, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
                            except Exception:
                                pass
                        elif z.zone_type == "tripwire":
                            try:
                                p1 = self.transformer.world_to_image(*z.world_coords[0])
                                p2 = self.transformer.world_to_image(*z.world_coords[1])
                                ip1 = (int(p1[0]), int(p1[1]))
                                ip2 = (int(p2[0]), int(p2[1]))
                                cv2.line(frame, ip1, ip2, (0, 0, 255), 2, cv2.LINE_AA)
                                cv2.putText(frame, z.name, (ip1[0] + 10, ip1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
                            except Exception:
                                pass

                # 2. YOLOv8 Detection
                results = self.detector(clean_frame, classes=[0], conf=min(self.conf_threshold, self.tracker.low_conf_threshold), verbose=False)
                detections = []
                for r in results:
                    if r.boxes is not None:
                        for idx_b, box in enumerate(r.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            detections.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": conf,
                                "foot_point": ((x1 + x2) / 2.0, float(y2)),
                                "keypoints": r.keypoints.data[idx_b].cpu().numpy() if self.show_pose and r.keypoints is not None and idx_b < len(r.keypoints.data) else None
                            })

                # 3. Kalman Tracker & Homography
                active_tracks = self.tracker.update(detections, homography_transformer=self.transformer,
                                                    pose_classifier=self.pose_classifier if self.show_pose else None,
                                                    zones=self.zones, elapsed_dt=elapsed_dt)

                # 4. Threat Kinetics & Dual-View Annotation
                alarm_states, _ = self.monitor.update(active_tracks, self.zones, elapsed_dt, speed_alarm=self.speed_alarm_kmh)
                radar_tracks = []
                alarm_triggered = False

                for trk in active_tracks:
                    w_pos = trk.get("world_pos")
                    is_breach = alarm_states[trk['track_id']]['is_breaching']
                    alarm_triggered |= is_breach
                    speed_kmh = float(trk.get('speed', 0))
                    if self.show_pose and trk.get('keypoints') is not None:
                        draw_pose_skeleton(frame, trk['keypoints'])

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

                    tag = f"#{trk['track_id']} | " + (f"{speed_kmh:.1f} km/h" if w_pos is not None else "speed unavailable")
                    cv2.putText(frame, tag, (x1, max(16, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

                # 5. Render BEV Radar
                radar_canvas = self.radar.render(self.zones if self.transformer.is_calibrated else [], radar_tracks,
                                                 title="GROUND RADAR" if self.transformer.is_calibrated else "UNCALIBRATED - METRIC ALARMS DISABLED")

                # 6. Composite Dual-View
                dual_view = BEVRadarRenderer.create_dual_view(frame, radar_canvas)
                dh, dw = dual_view.shape[:2]

                # 7. Render Telemetry HUD Bar
                cv2.rectangle(dual_view, (0, 0), (dw, 40), (12, 16, 20), -1)
                feed_label = f"FEED: [{self.current_video_key}] {Path(self.videos[self.current_video_key]).name} | FPS: {fps:.1f} | INTRUDERS: {len(active_tracks)}"
                cv2.putText(dual_view, feed_label, (16, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 240, 255), 1, cv2.LINE_AA)

                if alarm_triggered:
                    self._trigger_audio_alarm()
                    cv2.putText(dual_view, "⚠️ PERIMETER BREACH ACTIVE", (dw // 2 - 120, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                audio_status = "ON" if getattr(self, "audio_alarm_enabled", True) else "MUTED"
                controls_str = f"[1-5/0] Switch Video | [SPACE] Pause | [G] Grid | [M] Audio: {audio_status} | [S] Snapshot | [Q] Quit"
                (cw, _), _ = cv2.getTextSize(controls_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.putText(dual_view, controls_str, (dw - cw - 16, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 180, 200), 1, cv2.LINE_AA)

                last_view = dual_view
                cv2.imshow(window_name, dual_view)
                key = cv2.waitKey(20 if not self.paused else 100) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.paused = not self.paused
                elif key == ord('g'):
                    self.show_grid = not self.show_grid
                elif key == ord('m'):
                    self.audio_alarm_enabled = not getattr(self, "audio_alarm_enabled", True)
                    if hasattr(self, "siren_controller") and self.siren_controller is not None:
                        self.siren_controller.set_enabled(self.audio_alarm_enabled)
                    print(f"🔊 Audio Siren Alarm: {'ENABLED' if self.audio_alarm_enabled else 'MUTED'}")
                elif chr(key) in self.videos:
                    self._load_source(chr(key))
                elif key == ord('s'):
                    snap_path = f"data/snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(snap_path, dual_view)
                    print(f"✅ Saved snapshot to {snap_path}")

        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = TripwireInteractiveDemo()
    demo.run()
