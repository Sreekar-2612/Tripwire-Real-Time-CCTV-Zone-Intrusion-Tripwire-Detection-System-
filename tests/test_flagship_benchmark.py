import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import get_default_zones
from cv_core.tracking.tracker import MultiTargetTracker, SingleTrack
from cv_core.monitoring import ZoneMonitor
from ultralytics import YOLO

FLAGSHIP_VIDEO = Path("data/videos/flagship_multi_threat_benchmark.mp4")
FLAGSHIP_CALIB = Path("data/videos/flagship_multi_threat_benchmark.mp4.calibration.json")

def test_flagship_media_and_calibration_files_exist():
    """Verify flagship benchmark media and its dedicated camera calibration exist."""
    assert FLAGSHIP_VIDEO.exists(), f"Missing {FLAGSHIP_VIDEO}"
    assert FLAGSHIP_CALIB.exists(), f"Missing {FLAGSHIP_CALIB}"
    assert FLAGSHIP_VIDEO.stat().st_size > 1_000_000, "Video size should be > 1MB"

def test_flagship_video_properties():
    """Verify video is 720p HD (1280x720) @ 20 FPS, 240 frames."""
    SingleTrack.reset_id_counter(1)
    cap = cv2.VideoCapture(str(FLAGSHIP_VIDEO))
    try:
        assert cap.isOpened(), "Could not open flagship video"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(cap.get(cv2.CAP_PROP_FPS), 1)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert (w, h) == (1280, 720), f"Expected 1280x720 HD, got {w}x{h}"
        assert fps == 20.0, f"Expected 20.0 FPS, got {fps}"
        assert frames == 240, f"Expected 240 frames, got {frames}"
    finally:
        cap.release()

def test_flagship_calibration_binding():
    """Verify homography calibration correctly auto-binds to the flagship feed."""
    transformer = HomographyTransformer.for_source(str(FLAGSHIP_VIDEO), (1280, 720))
    assert transformer.is_calibrated, "Flagship homography must auto-load successfully"
    assert transformer.H is not None
    assert transformer.H.shape == (3, 3)
    assert transformer.unit == "meters"

def test_flagship_calibrated_zones():
    """Verify calibrated security zones for the flagship template."""
    zones = get_default_zones(str(FLAGSHIP_VIDEO))
    zone_ids = [z.zone_id for z in zones]
    assert "z_vault" in zone_ids, "Expected 'z_vault' polygon zone"
    assert "tw_chk" in zone_ids, "Expected 'tw_chk' tripwire line"

    vault = next(z for z in zones if z.zone_id == "z_vault")
    wire = next(z for z in zones if z.zone_id == "tw_chk")

    assert vault.zone_type == "polygon"
    assert vault.area_m2 > 10.0, "Vault polygon must have physical ground area"
    assert wire.zone_type == "tripwire"
    assert wire.perimeter_length_m >= 3.0, "Checkpoint tripwire must span corridor"

def test_flagship_cv_pipeline_threat_scenarios():
    """
    Verify complete multi-threat benchmark detection:
      - Target #1: Authorized Worker (green box, nominal speed < 4 km/h, 0 false alarms)
      - Target #2: Tripwire Intruder (CROSS_IN on CHECKPOINT TRIPWIRE)
      - Target #3: Loiterer (ENTER + LOITER on RESTRICTED VAULT)
      - Target #4: Fleeing Intruder (SPEED_START with speed > 14 km/h)
    """
    SingleTrack.reset_id_counter(1)
    cap = cv2.VideoCapture(str(FLAGSHIP_VIDEO))
    transformer = HomographyTransformer.for_source(str(FLAGSHIP_VIDEO), (1280, 720))
    zones = get_default_zones(str(FLAGSHIP_VIDEO))
    tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
    monitor = ZoneMonitor()
    detector = YOLO("yolov8n.pt")

    all_events = []
    tracks_seen = set()
    max_speeds = {}
    dt = 1.0 / 20.0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = detector(frame, classes=[0], verbose=False)[0]
        dets = [{"bbox": [float(x) for x in b.xyxy[0]], "confidence": float(b.conf[0])} for b in res.boxes]
        tracks = tracker.update(dets, elapsed_dt=dt, homography_transformer=transformer)
        for t in tracks:
            tid = t["track_id"]
            tracks_seen.add(tid)
            max_speeds[tid] = max(max_speeds.get(tid, 0.0), t.get("speed", 0.0))
        states, events = monitor.update(tracks, zones, dt=dt, dwell_confirm_sec=1.5, speed_alarm=5.0)
        for e in events:
            all_events.append((frame_idx, e))
        frame_idx += 1
    cap.release()

    assert frame_idx == 240, f"Processed {frame_idx}/240 frames"
    assert len(tracks_seen) == 4, f"Expected exactly 4 unique tracks, got: {tracks_seen}"

    # Target #4: Rapid fleeing intruder (highest speed track)
    t4_id = max(max_speeds, key=max_speeds.get)
    assert max_speeds[t4_id] >= 14.0, f"Target #4 speed should be >= 14 km/h, got: {max_speeds[t4_id]}"
    t4_speed = [e for _, e in all_events if e["event"] == "SPEED_START" and e["track_id"] == t4_id]
    assert len(t4_speed) >= 1, "Fleeing intruder must trigger speed alarm (SPEED_START)"

    # Target #2: Tripwire breach
    t2_cross = [e for _, e in all_events if e["event"] == "CROSS_IN" and e["zone_id"] == "tw_chk"]
    assert len(t2_cross) >= 1, "Tripwire intruder must trigger tripwire crossing (CROSS_IN)"
    t2_id = t2_cross[0]["track_id"]

    # Target #3: Loitering in vault
    t3_loiter = [e for _, e in all_events if e["event"] == "LOITER" and e["zone_id"] == "z_vault"]
    assert len(t3_loiter) >= 1, "Loiterer must trigger vault loiter alarm (LOITER)"
    t3_id = t3_loiter[0]["track_id"]

    # Target #1: Authorized worker (remaining track with 0 breach events)
    auth_candidates = tracks_seen - {t2_id, t3_id, t4_id}
    assert len(auth_candidates) == 1, f"Expected exactly 1 authorized worker, got: {auth_candidates}"
    t1_id = auth_candidates.pop()
    t1_events = [e for _, e in all_events if e["track_id"] == t1_id]
    assert len(t1_events) == 0, f"Target #1 should have zero breach alarms, got: {t1_events}"
    assert max_speeds.get(t1_id, 0.0) < 4.0, f"Target #1 speed should be nominal, got: {max_speeds.get(t1_id)}"
