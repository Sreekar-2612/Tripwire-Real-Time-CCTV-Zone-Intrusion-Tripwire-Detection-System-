import argparse
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

def run_cv_engine(
    video_source: str,
    calibration_file: str = None,
    output_video: str = None,
    show_window: bool = True
):
    print("="*75)
    print("🚀 TRIPWIRE COMPUTER VISION ENGINE — BEV RADAR & KINETIC SURVEILLANCE")
    print("="*75)

    # 1. Initialize Homography Transformer
    transformer = HomographyTransformer()
    if calibration_file and Path(calibration_file).exists():
        transformer = HomographyTransformer.load_calibration(calibration_file)
        print(f"✅ Loaded Ground Homography Calibration from '{calibration_file}'")
    elif Path("data/calibration.json").exists():
        transformer = HomographyTransformer.load_calibration("data/calibration.json")
        print("✅ Loaded default Ground Homography Calibration from 'data/calibration.json'")
    else:
        print("ℹ️ No calibration specified. Using default ground bounds (20m x 20m).")
        # Default identity mapping for demonstration
        transformer.set_homography(np.eye(3))

    # 2. Initialize Models & Tracker
    print("Loading YOLOv8 Detector & Tracker...")
    detector = YOLO("yolov8n.pt")
    tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
    pose_classifier = PoseKineticClassifier()
    radar = BEVRadarRenderer(radar_size=480, world_extent_m=20.0, grid_step_m=2.0)

    # Example Metric Ground Zones (in meters)
    ground_zones = [
        MetricZone3D("z1", "SECURE ZONE", "polygon", [(3.0, 3.0), (12.0, 3.0), (12.0, 10.0), (3.0, 10.0)]),
        MetricZone3D("tw1", "PERIMETER WIRE", "tripwire", [(0.0, 6.0), (15.0, 6.0)])
    ]

    # Open Video Capture
    cap_src = int(video_source) if str(video_source).isdigit() else video_source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    writer = None

    frame_count = 0
    t_start = time.time()

    print("\nProcessing frames... Press 'q' in preview window to exit.\n")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_count += 1

            # 1. YOLO Person Detection
            results = detector(frame, classes=[0], conf=0.35, verbose=False)
            detections = []

            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        foot_x = (x1 + x2) / 2.0
                        foot_y = y2
                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "foot_point": (float(foot_x), float(foot_y))
                        })

            # 2. Multi-Target Tracking & Homography Projection
            active_tracks = tracker.update(detections, homography_transformer=transformer)

            # 3. Zone Breaches & Pose Kinetics
            radar_tracks = []
            for trk in active_tracks:
                w_pos = trk.get("world_pos")
                is_breaching = False
                if w_pos:
                    for z in ground_zones:
                        if z.zone_type == "tripwire":
                            world_trail = trk.get("world_trail", [])
                            if len(world_trail) >= 2:
                                crossing = z.check_tripwire_crossing(world_trail[-2], world_trail[-1])
                                if crossing is not None:
                                    is_breaching = True
                                    break
                        elif z.contains_point(w_pos[0], w_pos[1]):
                            is_breaching = True
                            break

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
                tag = f"#{trk['track_id']} | {speed_kmh:.1f} km/h"
                cv2.putText(frame, tag, (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

            # 4. Render BEV Ground Radar Display
            radar_canvas = radar.render(ground_zones, radar_tracks)

            # 5. Composite Dual-View (Perspective + Radar)
            dual_view = BEVRadarRenderer.create_dual_view(frame, radar_canvas)

            # Initialize writer lazily with exact matching dual-view dimensions
            if output_video and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                dh, dw = dual_view.shape[:2]
                writer = cv2.VideoWriter(output_video, fourcc, fps, (dw, dh))
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
        cv2.destroyAllWindows()
        print(f"\n✅ Processing complete: {frame_count} frames processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tripwire CV Pipeline Runner")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or webcam index (e.g. 0)")
    parser.add_argument("--calibration", type=str, default=None, help="Path to calibration.json")
    parser.add_argument("--output", type=str, default=None, help="Path to save dual-view output MP4")
    parser.add_argument("--no-preview", action="store_true", help="Disable GUI preview window")
    args = parser.parse_args()

    run_cv_engine(
        video_source=args.source,
        calibration_file=args.calibration,
        output_video=args.output,
        show_window=not args.no_preview
    )
