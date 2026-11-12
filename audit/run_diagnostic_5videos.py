import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np

# Add project root
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from ultralytics import YOLO
from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import get_default_zones
from cv_core.tracking.tracker import MultiTargetTracker
from cv_core.kinetics.pose_classifier import PoseKineticClassifier
from cv_core.video import FrameClock
from cv_core.monitoring import ZoneMonitor
from cv_core.visualizer.bev_radar import BEVRadarRenderer

videos = [
    'data/videos/corridor_tripwire_breach.mp4',
    'data/videos/high_speed_fleeing_breach.mp4',
    'data/videos/walking_cctv.mp4',
    'data/videos/cctv_sample.mp4',
    'data/videos/solutions_ci_demo.mp4'
]

results = {}

for v_path in videos:
    p = Path(v_path)
    print(f"\n--- Processing {p.name} ---")
    cap = cv2.VideoCapture(str(p))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clock = FrameClock(cap, live=False)
    # Load calibration: check explicit or default
    transformer = HomographyTransformer.for_source(str(p), (w, h))
    if not transformer.is_calibrated and Path("data/calibration.json").exists():
        transformer = HomographyTransformer.load_calibration("data/calibration.json")

    detector = YOLO("yolov8n-pose.pt")
    tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
    pose_classifier = PoseKineticClassifier()
    monitor = ZoneMonitor()
    zones = get_default_zones(str(p))

    processed_frames = 0
    all_events = []
    track_ids_seen = set()
    max_speed = 0.0
    postures_seen = set()
    t_start = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        dt = clock.tick()
        processed_frames += 1

        res = detector(frame, classes=[0], conf=0.25, imgsz=480, verbose=False)[0]
        detections = []
        if res.boxes is not None:
            for idx, box in enumerate(res.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                foot_x = (x1 + x2) / 2.0
                foot_y = y2
                kpts = res.keypoints.data[idx].cpu().numpy() if res.keypoints is not None else None
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "foot_point": (float(foot_x), float(foot_y)),
                    "keypoints": kpts
                })

        tracks = tracker.update(
            detections,
            homography_transformer=transformer,
            pose_classifier=pose_classifier,
            zones=zones,
            elapsed_dt=dt
        )

        states, events = monitor.update(tracks, zones, dt, dwell_confirm_sec=1.0, speed_alarm=5.0)
        all_events.extend(events)

        for trk in tracks:
            tid = trk['track_id']
            track_ids_seen.add(tid)
            spd = trk.get('speed', 0.0)
            if spd > max_speed:
                max_speed = spd
            postures_seen.add(trk.get('posture', 'standing'))

    t_end = time.perf_counter()
    cap.release()
    elapsed = t_end - t_start

    event_counts = {}
    for ev in all_events:
        ename = ev['event']
        event_counts[ename] = event_counts.get(ename, 0) + 1

    results[p.name] = {
        "input_resolution": f"{w}x{h}",
        "source_fps": src_fps,
        "input_frames": total_frames,
        "processed_frames": processed_frames,
        "elapsed_sec": round(elapsed, 2),
        "processing_fps": round(processed_frames / max(0.001, elapsed), 2),
        "calibrated": transformer.is_calibrated,
        "unique_tracks": len(track_ids_seen),
        "track_ids": sorted(list(track_ids_seen)),
        "max_speed_kmh": round(max_speed, 2),
        "postures_seen": sorted(list(postures_seen)),
        "event_summary": event_counts,
        "events_sample": all_events[:10]
    }
    print(f"Done {p.name}: {processed_frames}/{total_frames} frames in {elapsed:.2f}s ({processed_frames/elapsed:.2f} FPS). Tracks: {len(track_ids_seen)}, Events: {event_counts}")

with open("audit/video_feed_diagnostics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll 5 video feeds successfully audited and saved to audit/video_feed_diagnostics.json!")
