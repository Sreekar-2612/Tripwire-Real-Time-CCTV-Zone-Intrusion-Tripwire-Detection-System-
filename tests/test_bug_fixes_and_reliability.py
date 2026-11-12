import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

from cv_core.tracking.kalman_filter import KalmanFilter2D
from cv_core.tracking.tracker import MultiTargetTracker, SingleTrack, TrackState, compute_iou_matrix
from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D
from cv_core.visualizer.bev_radar import BEVRadarRenderer
from cv_core.models.benchmark import InferenceProfiler
from cv_core.kinetics.pose_classifier import PoseKineticClassifier, PostureType


# =============================================================================
# 1. EMPTY DETECTIONS & TARGET DEPARTURES
# =============================================================================

def test_tracker_empty_and_none_detections():
    """Verify that empty, None, or malformed detections lists do not crash tracker."""
    tracker = MultiTargetTracker(min_hits=2, max_age=5)

    # Empty list on empty tracker
    out_empty = tracker.update([])
    assert out_empty == []

    # None on empty tracker
    out_none = tracker.update(None)
    assert out_none == []

    # Malformed list with None and invalid dicts
    malformed = [None, {}, {"other": 123}, {"bbox": None}, {"bbox": [10.0, 10.0]}]
    out_malformed = tracker.update(malformed)
    assert out_malformed == []
    assert len(tracker.tracks) == 0


def test_tracker_single_frame_detection_and_departure():
    """Verify single-frame transient detections transition to LOST and expire after max_age."""
    tracker = MultiTargetTracker(min_hits=2, max_age=3)

    # Frame 1: Detection appears (spawns TENTATIVE track, hits=1)
    det1 = [{"bbox": [100.0, 100.0, 150.0, 200.0], "confidence": 0.9, "foot_point": (125.0, 200.0)}]
    out1 = tracker.update(det1)
    assert len(out1) == 0  # Requires min_hits=2
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].state == TrackState.TENTATIVE

    # Target immediately departs: frame 2 empty
    out2 = tracker.update([])
    assert len(out2) == 0
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].state == TrackState.LOST

    # Exceed max_age (frames 3, 4, 5)
    for _ in range(3):
        tracker.update([])

    # Track must now be pruned (DELETED)
    assert len(tracker.tracks) == 0


def test_tracker_confirmed_target_departure_and_pruning():
    """Verify confirmed track departs, goes to LOST, and cleanly deletes after max_age."""
    tracker = MultiTargetTracker(min_hits=2, max_age=4)

    # Frame 1 & 2: Target confirmed
    det = [{"bbox": [50.0, 50.0, 100.0, 100.0], "confidence": 0.9, "foot_point": (75.0, 100.0)}]
    tracker.update(det)
    out2 = tracker.update(det)
    assert len(out2) == 1
    assert tracker.tracks[0].state == TrackState.CONFIRMED

    # Target departs
    for _ in range(4):
        tracker.update([])
        assert tracker.tracks[0].state == TrackState.LOST

    # Next missing frame exceeds max_age -> deleted
    tracker.update([])
    assert len(tracker.tracks) == 0


# =============================================================================
# 2. TARGETS WITH ZERO VELOCITY OR NAN / INF COORDINATES
# =============================================================================

def test_tracker_and_kalman_stationary_zero_velocity():
    """Verify that a stationary target has exactly 0.0 velocity and zero speed."""
    tracker = MultiTargetTracker(min_hits=1, max_age=5)
    det = [{"bbox": [100.0, 100.0, 150.0, 200.0], "confidence": 0.9, "foot_point": (125.0, 200.0)}]

    # Feed identical detection across 5 frames
    for _ in range(5):
        out = tracker.update(det)

    assert len(out) == 1
    assert np.isclose(out[0]["speed"], 0.0, atol=0.2)
    assert np.isclose(out[0]["vx"], 0.0, atol=0.1)
    assert np.isclose(out[0]["vy"], 0.0, atol=0.1)


def test_tracker_nan_and_inf_coordinate_resilience():
    """Verify tracker gracefully ignores NaN and Inf coordinates in detections."""
    tracker = MultiTargetTracker(min_hits=1, max_age=5)

    # Valid detection first
    tracker.update([{"bbox": [100.0, 100.0, 150.0, 200.0], "confidence": 0.9, "foot_point": (125.0, 200.0)}])
    assert len(tracker.tracks) == 1

    # Detections containing NaN and Inf coordinates
    corrupt_dets = [
        {"bbox": [np.nan, 100.0, 150.0, 200.0], "confidence": 0.9, "foot_point": (125.0, 200.0)},
        {"bbox": [100.0, np.nan, 150.0, 200.0], "confidence": 0.9, "foot_point": (125.0, 200.0)},
        {"bbox": [100.0, 100.0, np.inf, 200.0], "confidence": 0.9, "foot_point": (125.0, 200.0)},
        {"bbox": [100.0, 100.0, 150.0, 200.0], "confidence": 0.9, "foot_point": (np.nan, np.nan)},
        {"bbox": [100.0, 100.0, 150.0, 200.0], "confidence": np.nan, "foot_point": (125.0, 200.0)}
    ]

    # Tracker should process without unhandled exception
    out = tracker.update(corrupt_dets)
    # NaN bbox detections are discarded; NaN foot_point is recovered from valid bbox; NaN confidence defaulted to 0.0
    for trk in tracker.tracks:
        assert np.all(np.isfinite(trk.bbox))
        assert np.all(np.isfinite(trk.foot_point))


def test_kalman_filter_nan_inf_immunity():
    """Verify KalmanFilter2D does not produce NaNs or crash when fed non-finite inputs."""
    kf = KalmanFilter2D(dt=0.1)
    kf.initialize(np.nan, 10.0)
    assert np.all(np.isfinite(kf.position))

    kf.update(np.nan, np.nan)
    assert np.all(np.isfinite(kf.position))
    assert np.all(np.isfinite(kf.velocity))

    kf.update(np.inf, 5.0)
    assert np.all(np.isfinite(kf.position))

    pred_x, pred_y = kf.predict()
    assert np.isfinite(pred_x) and np.isfinite(pred_y)


def test_bev_radar_nan_and_zero_velocity_handling():
    """Verify BEVRadarRenderer handles NaN positions, zero velocity, and zero height safely."""
    renderer = BEVRadarRenderer(radar_size=400, world_extent_m=20.0)

    # Test world_to_radar_pixel with NaN
    px, py = renderer.world_to_radar_pixel(np.nan, np.nan)
    assert isinstance(px, int) and isinstance(py, int)

    # Test rendering track with zero velocity and track with NaN velocity
    tracks = [
        {
            "track_id": 1,
            "world_pos": (5.0, 5.0),
            "world_trail": [(5.0, 5.0), (5.0, 5.0)],
            "speed_kmh": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_breaching": False
        },
        {
            "track_id": 2,
            "world_pos": (np.nan, np.nan),  # Should be skipped safely
            "world_trail": [],
            "speed_kmh": np.nan,
            "vx": np.nan,
            "vy": np.nan,
            "is_breaching": False
        }
    ]
    radar_img = renderer.render([], tracks)
    assert radar_img.shape == (400, 400, 3)

    # Test create_dual_view with zero height or empty frames
    dummy_cam = np.zeros((480, 640, 3), dtype=np.uint8)
    empty_cam = np.zeros((0, 0, 3), dtype=np.uint8)
    dual = BEVRadarRenderer.create_dual_view(dummy_cam, radar_img)
    assert dual.shape[0] == 480
    assert BEVRadarRenderer.create_dual_view(empty_cam, radar_img).shape == (400, 400, 3)


# =============================================================================
# 3. HOMOGRAPHY SINGULARITY & POINTS BEHIND CAMERA PLANE
# =============================================================================

def test_homography_nan_and_inf_inputs():
    """Verify image_to_world and world_to_image reject non-finite inputs."""
    H = np.eye(3, dtype=np.float64)
    transformer = HomographyTransformer(H_matrix=H)

    with pytest.raises(ValueError):
        transformer.image_to_world(np.nan, 10.0)

    with pytest.raises(ValueError):
        transformer.world_to_image(5.0, np.nan)


def test_homography_points_behind_camera_plane():
    """Verify detection of points behind the camera plane and horizon."""
    # Calibration mapping ground corridor
    img_pts = [(300.0, 150.0), (500.0, 150.0), (650.0, 500.0), (150.0, 500.0)]
    world_pts = [(0.0, 8.0), (4.0, 8.0), (4.0, 0.0), (0.0, 0.0)]
    transformer = HomographyTransformer.from_point_correspondences(img_pts, world_pts)

    # In front of camera (positive Y on corridor floor)
    assert transformer.is_world_point_in_front_of_camera(2.0, 4.0)

    # Point behind camera plane (e.g. far behind camera Y = -10.0)
    assert not transformer.is_world_point_in_front_of_camera(0.0, -10.0)

    # Strict check raises ValueError for point behind camera
    with pytest.raises(ValueError, match="behind camera plane"):
        transformer.world_to_image(0.0, -10.0, check_camera_plane=True)

    # Default check still computes projection without crashing
    u, v = transformer.world_to_image(0.0, -10.0, check_camera_plane=False)
    assert np.isfinite(u) and np.isfinite(v)


def test_homography_inverse_projective_singularity():
    """Verify inverse projection singularity raises ZeroDivisionError."""
    # Construct an H matrix where H_inv produces denominator = 0
    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, -5.0]
    ])
    transformer = HomographyTransformer(H_matrix=H)
    # Inverse singularity occurs when reprojected[2] == 0 (at Y=1.0)
    with pytest.raises(ZeroDivisionError):
        transformer.world_to_image(0.0, 1.0)


# =============================================================================
# 4. DIVISION BY ZERO IN FPS & VELOCITY CALCULATORS
# =============================================================================

def test_inference_profiler_zero_runs_division_by_zero():
    """Verify InferenceProfiler with test_runs=0 does not divide by zero."""
    profiler = InferenceProfiler(warmup_runs=0, test_runs=0)
    stats = profiler.profile_dummy_pipeline(imgsz=160)

    assert stats["num_runs"] == 0
    assert stats["fps"] == 0.0
    assert stats["total_latency_mean_ms"] == 0.0


def test_homography_velocity_division_by_zero():
    """Verify compute_velocity handles dt=0, zero fps, negative fps, and NaN timestamps."""
    transformer = HomographyTransformer()

    # Empty trajectory
    assert transformer.compute_velocity([])["speed_mps"] == 0.0

    # Single point
    assert transformer.compute_velocity([(1.0, 2.0)])["speed_mps"] == 0.0

    # Two points with identical timestamps (dt = 0)
    t_zero = [(1.0, 2.0, 10.0), (3.0, 4.0, 10.0)]
    assert transformer.compute_velocity(t_zero)["speed_mps"] == 0.0

    # Trajectory with zero or negative fps
    t_no_ts = [(1.0, 2.0), (3.0, 4.0)]
    res_zero_fps = transformer.compute_velocity(t_no_ts, fps=0.0)
    assert np.isfinite(res_zero_fps["speed_mps"])
    assert res_zero_fps["speed_mps"] > 0.0

    # Trajectory with NaN coordinates
    t_nan = [(np.nan, 2.0, 1.0), (3.0, 4.0, 2.0)]
    assert transformer.compute_velocity(t_nan)["speed_mps"] == 0.0

    # Trajectory with stationary coordinates (zero velocity)
    t_stationary = [(5.0, 5.0, 1.0), (5.0, 5.0, 2.0)]
    res_stat = transformer.compute_velocity(t_stationary)
    assert res_stat["speed_mps"] == 0.0
    assert res_stat["speed_kmh"] == 0.0
    assert res_stat["direction_deg"] == 0.0


# =============================================================================
# 5. METRIC ZONES & POSE CLASSIFIER DEFENSIVE CHECKS
# =============================================================================

def test_metric_zone_nan_safety():
    """Verify MetricZone3D handles NaN coordinates safely."""
    zone = MetricZone3D("z", "Safe Zone", "polygon", [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    assert not zone.contains_point(np.nan, 5.0)
    assert not zone.contains_point(5.0, np.nan)

    wire = MetricZone3D("tw", "Fence", "tripwire", [(0.0, 5.0), (10.0, 5.0)])
    assert wire.check_tripwire_crossing((np.nan, 2.0), (5.0, 8.0)) is None
    assert wire.check_tripwire_crossing((2.0, 2.0), (np.nan, np.nan)) is None


def test_pose_classifier_corrupt_keypoints_safety():
    """Verify PoseKineticClassifier handles None, empty, NaN, and 1D keypoint inputs."""
    classifier = PoseKineticClassifier()

    assert classifier.classify_pose(None)["posture"] == PostureType.UNKNOWN.value
    assert classifier.classify_pose(np.zeros((5, 2)))["posture"] == PostureType.UNKNOWN.value
    assert classifier.classify_pose(np.zeros((17, 1)))["posture"] == PostureType.UNKNOWN.value

    # 17 keypoints all NaN
    kpts_nan = np.full((17, 3), np.nan, dtype=np.float32)
    assert classifier.classify_pose(kpts_nan)["posture"] == PostureType.UNKNOWN.value

    # NaN speed
    kpts_valid = np.zeros((17, 3), dtype=np.float32)
    kpts_valid[:, 2] = 0.9
    kpts_valid[PoseKineticClassifier.LEFT_SHOULDER] = [100.0, 100.0, 0.9]
    kpts_valid[PoseKineticClassifier.RIGHT_SHOULDER] = [120.0, 100.0, 0.9]
    kpts_valid[PoseKineticClassifier.LEFT_HIP] = [100.0, 200.0, 0.9]
    kpts_valid[PoseKineticClassifier.RIGHT_HIP] = [120.0, 200.0, 0.9]
    res = classifier.classify_pose(kpts_valid, speed_mps=np.nan)
    assert res["posture"] == PostureType.STANDING.value


# =============================================================================
# 6. UNHANDLED CAMERA DISCONNECTS IN DEMO ENGINES
# =============================================================================

def test_web_stream_engine_camera_disconnect_fallback():
    """Verify WebStreamEngine returns a valid fallback JPEG image and does not crash on disconnect."""
    from scripts.web_demo import WebStreamEngine

    engine = WebStreamEngine()

    # Mock VideoCapture to simulate broken/disconnected camera
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)  # Disconnected camera read
    engine.cap = mock_cap

    # Call get_frame when camera is disconnected
    frame_bytes = engine.get_frame()

    assert len(frame_bytes) > 0  # Fallback warning frame generated!
    assert frame_bytes[:2] == b'\xff\xd8'  # Valid JPEG header!
    assert engine.cap is None  # Released disconnected capture


def test_web_stream_engine_reconnect_cooldown():
    """Verify WebStreamEngine avoids hammering VideoCapture with a 2-second reconnect cooldown."""
    from scripts.web_demo import WebStreamEngine
    import time

    engine = WebStreamEngine()
    engine.cap = None
    engine.last_reconnect_time = time.time()  # Recent attempt

    with patch.object(engine, "_init_cap") as mock_init:
        # Should not call _init_cap immediately due to 2.0s cooldown
        frame = engine.get_frame()
        assert len(frame) > 0
        mock_init.assert_not_called()


# =============================================================================
# 7. RESILIENT MULTI-BACKEND WEBCAM INITIALIZATION & FALLBACK
# =============================================================================

def test_open_capture_source_file_success(tmp_path):
    """Verify open_capture_source opens a valid video file cleanly."""
    from app_streamlit import open_capture_source
    test_video = tmp_path / "dummy.mp4"
    test_video.write_bytes(b"dummy video data")

    with patch("cv2.VideoCapture") as mock_vc:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_vc.return_value = mock_instance

        cap, src, backend, err = open_capture_source(str(test_video))
        assert cap is not None
        assert src == str(test_video)
        assert backend == "File"
        assert err is None


def test_open_capture_source_file_missing():
    """Verify open_capture_source returns descriptive error when file is missing."""
    from app_streamlit import open_capture_source
    with patch("cv2.VideoCapture") as mock_vc:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_vc.return_value = mock_instance

        cap, src, backend, err = open_capture_source("non_existent_file.mp4")
        assert cap is None
        assert "Could not open video file" in err


def test_open_capture_source_webcam_primary_success():
    """Verify open_capture_source validates warmup reads and sets 640x480 resolution."""
    from app_streamlit import open_capture_source
    with patch("cv2.VideoCapture") as mock_vc:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, dummy_frame)
        mock_vc.return_value = mock_instance

        cap, idx, backend, err = open_capture_source(0, max_retries=1)
        assert cap is not None
        assert idx == 0
        assert err is None
        mock_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
        mock_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def test_open_capture_source_webcam_fallback_to_zero():
    """Verify open_capture_source falls back to active camera index 0 if index 1 fails."""
    from app_streamlit import open_capture_source

    def mock_vc_constructor(idx, *args):
        m = MagicMock()
        if idx == 1:
            m.isOpened.return_value = False
        else:
            m.isOpened.return_value = True
            m.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        return m

    with patch("cv2.VideoCapture", side_effect=mock_vc_constructor):
        cap, actual_idx, backend, err = open_capture_source(1, max_retries=1)
        assert cap is not None
        assert actual_idx == 0
        assert err is None


def test_open_capture_source_webcam_all_locked():
    """Verify open_capture_source returns structured troubleshooting info when all backends fail."""
    from app_streamlit import open_capture_source
    with patch("cv2.VideoCapture") as mock_vc:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_vc.return_value = mock_instance

        cap, idx, backend, err = open_capture_source(0, max_retries=1)
        assert cap is None
        assert "Could not access webcam device index #0" in err
        assert "Windows Camera Privacy" in err or "Windows Settings" in err

