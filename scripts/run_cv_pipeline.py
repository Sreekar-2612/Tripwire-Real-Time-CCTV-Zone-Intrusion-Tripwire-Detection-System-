import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from ultralytics import YOLO
from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D
from cv_core.tracking.tracker import MultiTargetTracker
from cv_core.visualizer.bev_radar import BEVRadarRenderer
from cv_core.kinetics.pose_classifier import PoseKineticClassifier
from cv_core.video import FrameClock
from cv_core.monitoring import ZoneMonitor

def run_cv_engine(
    video_source: str,
    calibration_file: str = None,
    output_video: str = None,
    show_window: bool = True,
    use_pose: bool = False
):
    print("="*75)
    print("🚀 TRIPWIRE COMPUTER VISION ENGINE — BEV RADAR & KINETIC SURVEILLANCE")
    print("="*75)

    if output_video and Path(output_video).resolve() == Path(str(video_source)).resolve():
        raise ValueError('Output video must not overwrite the input')

    # 2. Initialize Models & Tracker
    print("Loading YOLOv8 Detector & Tracker...")
    detector = YOLO("yolov8n-pose.pt" if use_pose else "yolov8n.pt")
    tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
    pose_classifier = PoseKineticClassifier()
    radar = BEVRadarRenderer(radar_size=480, world_extent_m=20.0, grid_step_m=2.0)

    # Example Metric Ground Zones (in meters)
    ground_zones = [
        MetricZone3D("z1", "RESTRICTED VAULT", "polygon", [(3.5, 1.2), (7.5, 1.2), (7.5, 2.8), (3.5, 2.8)]),
        MetricZone3D("tw1", "PERIMETER FENCE", "tripwire", [(2.0, 2.2), (8.0, 2.2)])
    ]

    # Open Video Capture
    cap_src = int(video_source) if str(video_source).isdigit() else video_source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {video_source}")
        cap.release()
        raise RuntimeError(f"Could not open video source: {video_source}")

    clock = FrameClock(cap, live=str(video_source).isdigit() or "://" in str(video_source))
    fps = 1.0 / clock.interval
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    try:
        transformer = HomographyTransformer.for_source(video_source, (w, h), calibration_file)
        if not transformer.is_calibrated and calibration_file is None and Path("data/calibration.json").exists():
            transformer = HomographyTransformer.load_calibration("data/calibration.json")
    except Exception:
        cap.release()
        raise
    if not transformer.is_calibrated:
        print('Uncalibrated feed: metric speed and spatial alarms are disabled.')
    monitor = ZoneMonitor()
    writer = None

    frame_count = 0
    t_start = time.time()

    print("\nProcessing frames... Press 'q' in preview window to exit.\n")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            elapsed_dt = clock.tick()
            frame_count += 1

            # 1. YOLO Person Detection
            results = detector(frame, classes=[0], conf=tracker.low_conf_threshold, verbose=False)
            detections = []

            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for index, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        foot_x = (x1 + x2) / 2.0
                        foot_y = y2
                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "foot_point": (float(foot_x), float(foot_y)),
                            "keypoints": r.keypoints.data[index].cpu().numpy() if use_pose and r.keypoints is not None else None
                        })

            # 2. Multi-Target Tracking & Homography Projection
            active_tracks = tracker.update(
                detections,
                homography_transformer=transformer,
                pose_classifier=pose_classifier if use_pose else None,
                zones=ground_zones, elapsed_dt=elapsed_dt
            )

            # 3. Zone Breaches & Pose Kinetics
            alarm_states, _ = monitor.update(active_tracks, ground_zones, elapsed_dt)
            radar_tracks = []
            for trk in active_tracks:
                w_pos = trk.get("world_pos")
                is_breaching = alarm_states[trk['track_id']]['is_breaching']

                speed_kmh = float(trk.get("speed", 0.0))
                radar_tracks.append({
                    "track_id": trk["track_id"],
                    "world_pos": w_pos,
                    "world_trail": trk.get("world_trail", []),
                    "speed_kmh": speed_kmh,
                    "vx": trk.get("vx", 0.0),
                    "vy": trk.get("vy", 0.0),
                    "is_breaching": is_breaching
                })

                # Annotate perspective frame
                x1, y1, x2, y2 = map(int, trk["bbox"])
                box_color = (0, 0, 255) if is_breaching else (0, 230, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                tag = f"#{trk['track_id']} | " + (f"{speed_kmh:.1f} km/h" if w_pos is not None else "speed unavailable")
                cv2.putText(frame, tag, (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

            # 4. Render BEV Ground Radar Display
            radar_canvas = radar.render(ground_zones if transformer.is_calibrated else [], radar_tracks,
                                        title="GROUND RADAR" if transformer.is_calibrated else "UNCALIBRATED - METRIC ALARMS DISABLED")

            # 5. Composite Dual-View (Perspective + Radar)
            dual_view = BEVRadarRenderer.create_dual_view(frame, radar_canvas)

            # Initialize writer lazily with exact matching dual-view dimensions
            if output_video and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                dh, dw = dual_view.shape[:2]
                writer = cv2.VideoWriter(output_video, fourcc, fps, (dw, dh))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video: {output_video}")
                print(f"📹 Recording Dual-View output to: {output_video} ({dw}x{dh} @ {fps:.1f} FPS)")

            # Calculate FPS
            elapsed = time.time() - t_start
            cur_fps = frame_count / max(0.001, elapsed)
            cv2.putText(dual_view, f"FPS: {cur_fps:.1f} | Tracks: {len(active_tracks)}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if writer:
                writer.write(dual_view)

            if show_window:
                cv2.imshow("Tripwire CV Engine: Perspective (Left) + BEV Radar (Right)", dual_view)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        # Fix #15: only clean up windows when a preview was actually created
        if show_window:
            cv2.destroyAllWindows()
        print(f"\n✅ Processing complete: {frame_count} frames processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tripwire CV Pipeline Runner")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or webcam index (e.g. 0)")
    parser.add_argument("--calibration", type=str, default=None, help="Path to calibration.json")
    parser.add_argument("--output", type=str, default=None, help="Path to save dual-view output MP4")
    parser.add_argument("--no-preview", action="store_true", help="Disable GUI preview window")
    parser.add_argument("--pose", action="store_true", help="Use pose weights for action recognition")
    args = parser.parse_args()

    run_cv_engine(
        video_source=args.source,
        calibration_file=args.calibration,
        output_video=args.output,
        show_window=not args.no_preview,
        use_pose=args.pose
    )
