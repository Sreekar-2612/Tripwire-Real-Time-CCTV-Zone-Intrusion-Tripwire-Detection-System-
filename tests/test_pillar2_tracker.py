import pytest
import numpy as np

from cv_core.tracking.kalman_filter import KalmanFilter2D
from cv_core.tracking.tracker import MultiTargetTracker, SingleTrack, compute_iou_matrix, TrackState, SingleTrack

def test_kalman_filter_prediction_and_velocity_convergence():
    """Verify that Kalman filter accurately tracks linear motion and converges velocity."""
    np.random.seed(42)
    dt = 0.1
    # Moving along X axis at 10 units/s: x(t) = 10 * t
    kf = KalmanFilter2D(dt=dt, process_noise_std=0.1, measurement_noise_std=0.1)
    kf.initialize(0.0, 0.0)

    for i in range(1, 20):
        t = i * dt
        true_x = 10.0 * t
        true_y = 5.0
        
        # Add slight Gaussian measurement noise
        meas_x = true_x + np.random.normal(0, 0.05)
        meas_y = true_y + np.random.normal(0, 0.05)
        
        kf.predict()
        kf.update(meas_x, meas_y)

    est_x, est_y = kf.position
    est_vx, est_vy = kf.velocity

    assert np.isclose(est_x, 10.0 * 1.9, atol=0.5)
    assert np.isclose(est_y, 5.0, atol=0.5)
    assert np.isclose(est_vx, 10.0, atol=1.0)  # Converged to ~10 units/s
    assert np.isclose(est_vy, 0.0, atol=0.5)

def test_kalman_filter_coast_during_occlusion():
    """Verify that Kalman filter coasting (predict without update) propagates position correctly."""
    dt = 0.1
    kf = KalmanFilter2D(dt=dt)
    kf.initialize(10.0, 10.0)
    kf.x[2, 0] = 5.0  # vx = 5.0 units/s
    kf.x[3, 0] = 0.0

    # Predict for 5 steps without measurement updates (simulating occlusion)
    for _ in range(5):
        kf.predict()

    est_x, est_y = kf.position
    # Expected x = 10.0 + 5 * (0.1 * 5.0) = 12.5
    assert np.isclose(est_x, 12.5, atol=1e-3)

def test_iou_matrix_calculation():
    """Test IoU computation between overlapping and disjoint bounding boxes."""
    boxes_a = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 30.0, 30.0]
    ])
    boxes_b = np.array([
        [0.0, 0.0, 10.0, 10.0],    # Exact overlap with box 0: IoU = 1.0
        [5.0, 5.0, 15.0, 15.0],    # Partial overlap with box 0
        [100.0, 100.0, 110.0, 110.0]  # Disjoint
    ])

    iou_mat = compute_iou_matrix(boxes_a, boxes_b)
    assert iou_mat.shape == (2, 3)
    assert np.isclose(iou_mat[0, 0], 1.0, atol=1e-4)
    assert 0.1 < iou_mat[0, 1] < 0.3
    assert np.isclose(iou_mat[0, 2], 0.0, atol=1e-4)
    assert np.isclose(iou_mat[1, 2], 0.0, atol=1e-4)

def test_multi_target_tracker_continuous_tracking():
    """Test tracking a moving target across consecutive frames with ID preservation."""
    tracker = MultiTargetTracker(min_hits=2, max_age=5, iou_threshold=0.3)

    # Frame 1: Detection enters scene
    det_f1 = [{"bbox": [100.0, 100.0, 150.0, 200.0], "confidence": 0.9, "foot_point": (125.0, 200.0)}]
    out1 = tracker.update(det_f1)

    # Frame 2: Target moves slightly right
    det_f2 = [{"bbox": [105.0, 100.0, 155.0, 200.0], "confidence": 0.92, "foot_point": (130.0, 200.0)}]
    out2 = tracker.update(det_f2)

    assert len(out2) == 1
    track_id = out2[0]["track_id"]

    # Frame 3: Target moves further right
    det_f3 = [{"bbox": [110.0, 100.0, 160.0, 200.0], "confidence": 0.91, "foot_point": (135.0, 200.0)}]
    out3 = tracker.update(det_f3)

    assert len(out3) == 1
    assert out3[0]["track_id"] == track_id  # ID maintained!
    assert len(out3[0]["trail"]) >= 2

def test_tracker_track_expiry_after_max_age():
    """Verify that tracks are cleaned up after max_age missing frames."""
    tracker = MultiTargetTracker(min_hits=1, max_age=3)
    det = [{"bbox": [50.0, 50.0, 80.0, 100.0], "confidence": 0.85, "foot_point": (65.0, 100.0)}]
    
    # Register track
    out = tracker.update(det)
    assert len(out) == 1

    # Empty frames exceeding max_age
    for _ in range(5):
        tracker.update([])

    # Track should now be pruned
    assert len(tracker.tracks) == 0

def test_tracker_occlusion_coasting_and_recovery():
    """Verify that a track coasting across occlusion is recovered when detection reappears."""
    tracker = MultiTargetTracker(min_hits=2, max_age=5, iou_threshold=0.3)
    
    # Establish confirmed track with 2 hits moving rightward: 10 px per frame
    det1 = [{"bbox": [100.0, 100.0, 140.0, 180.0], "confidence": 0.9, "foot_point": (120.0, 180.0)}]
    tracker.update(det1)
    det2 = [{"bbox": [110.0, 100.0, 150.0, 180.0], "confidence": 0.9, "foot_point": (130.0, 180.0)}]
    out2 = tracker.update(det2)
    assert len(out2) == 1
    orig_id = out2[0]["track_id"]

    # Occlusion for 1 frame (empty detections)
    out_occ1 = tracker.update([])
    assert len(out_occ1) == 0  # not reported because state is LOST
    assert tracker.tracks[0].state == TrackState.LOST

    # Reappearance near Kalman predicted position (~140 px foot point)
    det_reappear = [{"bbox": [120.0, 100.0, 160.0, 180.0], "confidence": 0.88, "foot_point": (140.0, 180.0)}]
    out_rec = tracker.update(det_reappear)
    assert len(out_rec) == 1
    assert out_rec[0]["track_id"] == orig_id
    assert tracker.tracks[0].state == TrackState.CONFIRMED

def test_tracker_dt_parameter_and_velocity_scaling():
    """Verify that tracker dt correctly scales ground metric velocity."""
    trk = SingleTrack(bbox=[10.0, 10.0, 30.0, 30.0], confidence=0.9, foot_point=(20.0, 30.0), dt=0.05)
    # Update with world points 1.0 meter apart: dt=0.05s -> vx = 1.0 / 0.05 = 20.0 m/s = 72 km/h (capped at 35 km/h)
    trk.update(bbox=[10.0, 10.0, 30.0, 30.0], confidence=0.9, foot_point=(20.0, 30.0), world_pt=(0.0, 0.0))
    trk.update(bbox=[15.0, 10.0, 35.0, 30.0], confidence=0.9, foot_point=(25.0, 30.0), world_pt=(1.0, 0.0))
    assert np.isclose(trk.metric_vx, 20.0, atol=1.0)
    assert trk.metric_speed <= 35.0

def test_tracker_multi_target_cross_path_association():
    """Verify that Hungarian matching correctly distinguishes two targets passing each other."""
    tracker = MultiTargetTracker(min_hits=1, max_age=5, iou_threshold=0.2)
    
    # Target A at (60, 50), Target B at (200, 50)
    det_f1 = [
        {"bbox": [40.0, 30.0, 80.0, 70.0], "confidence": 0.9, "foot_point": (60.0, 70.0)},
        {"bbox": [180.0, 30.0, 220.0, 70.0], "confidence": 0.9, "foot_point": (200.0, 70.0)}
    ]
    out1 = tracker.update(det_f1)
    assert len(out1) == 2
    id_a, id_b = out1[0]["track_id"], out1[1]["track_id"]

    # Target A moves right to (70, 50), Target B moves left to (190, 50)
    det_f2 = [
        {"bbox": [50.0, 30.0, 90.0, 70.0], "confidence": 0.9, "foot_point": (70.0, 70.0)},
        {"bbox": [170.0, 30.0, 210.0, 70.0], "confidence": 0.9, "foot_point": (190.0, 70.0)}
    ]
    out2 = tracker.update(det_f2)
    assert len(out2) == 2
    out_ids = {t["track_id"] for t in out2}
    assert out_ids == {id_a, id_b}

def test_tracker_unmatched_detections_create_tentative_tracks():
    """Verify that min_hits=2 suppresses single-frame false-positive detections."""
    tracker = MultiTargetTracker(min_hits=2, max_age=5)
    det_ghost = [{"bbox": [50.0, 50.0, 80.0, 100.0], "confidence": 0.8, "foot_point": (65.0, 100.0)}]
    
    # First frame: track created but tentative, so output is empty
    out = tracker.update(det_ghost)
    assert len(out) == 0
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].state == TrackState.TENTATIVE

def test_bytetrack_two_stage_low_confidence_occlusion_recovery():
    """Verify ByteTrack Stage 2 matches low-confidence detections (0.15 <= conf < 0.50) to recover occluded target."""
    tracker = MultiTargetTracker(min_hits=2, max_age=5, iou_threshold=0.3, high_conf_threshold=0.50, low_conf_threshold=0.15)
    
    # Frame 1 & 2: Target established with high confidence
    det1 = [{"bbox": [100.0, 100.0, 150.0, 200.0], "confidence": 0.92, "foot_point": (125.0, 200.0)}]
    tracker.update(det1)
    det2 = [{"bbox": [105.0, 100.0, 155.0, 200.0], "confidence": 0.88, "foot_point": (130.0, 200.0)}]
    out2 = tracker.update(det2)
    assert len(out2) == 1
    track_id = out2[0]["track_id"]
    assert tracker.tracks[0].state == TrackState.CONFIRMED

    # Frame 3: Target suffers heavy occlusion / motion blur (conf drops to 0.28, low confidence)
    det_low = [{"bbox": [110.0, 100.0, 160.0, 200.0], "confidence": 0.28, "foot_point": (135.0, 200.0)}]
    out3 = tracker.update(det_low)
    
    # Stage 2 association should successfully maintain track identity
    assert len(out3) == 1
    assert out3[0]["track_id"] == track_id
    assert out3[0]["confidence"] == 0.28
    assert tracker.tracks[0].state == TrackState.CONFIRMED

def test_bytetrack_low_confidence_detections_never_initialize_new_tracks():
    """Verify that low-confidence detections (conf < 0.50) are discarded and never spawn ghost tracks."""
    tracker = MultiTargetTracker(min_hits=1, max_age=5, high_conf_threshold=0.50, low_conf_threshold=0.15)
    
    # Background noise detection with conf = 0.30
    det_noise = [{"bbox": [200.0, 200.0, 240.0, 280.0], "confidence": 0.30, "foot_point": (220.0, 280.0)}]
    out = tracker.update(det_noise)
    
    assert len(out) == 0
    assert len(tracker.tracks) == 0  # Discarded without track creation!

def test_ema_velocity_smoothing_eliminates_jitter():
    """Verify that Exponential Moving Average (alpha=0.35) effectively smooths out velocity jitter spikes."""
    trk = SingleTrack(
        bbox=[100.0, 100.0, 140.0, 180.0],
        confidence=0.9,
        foot_point=(120.0, 180.0),
        dt=0.1,
        ema_alpha=0.35
    )
    
    # Initial world point
    trk.update(bbox=[100.0, 100.0, 140.0, 180.0], confidence=0.9, foot_point=(120.0, 180.0), world_pt=(0.0, 0.0))
    # Frame 2: Step 0.2m in dt=0.1s -> raw_vx = 2.0 m/s
    trk.update(bbox=[102.0, 100.0, 142.0, 180.0], confidence=0.9, foot_point=(122.0, 180.0), world_pt=(0.2, 0.0))
    assert np.isclose(trk.metric_vx, 2.0, atol=1e-3)
    
    # Frame 3: Severe measurement jitter spike: jumps 1.0m in 0.1s -> raw_vx = 10.0 m/s
    trk.update(bbox=[110.0, 100.0, 150.0, 180.0], confidence=0.9, foot_point=(130.0, 180.0), world_pt=(1.2, 0.0))
    # Expected EMA: 0.35 * 10.0 + (1 - 0.35) * 2.0 = 3.5 + 1.3 = 4.8 m/s
    expected_ema_vx = 0.35 * 10.0 + 0.65 * 2.0
    assert np.isclose(trk.metric_vx, expected_ema_vx, atol=1e-3)
    # The jitter spike (10 m/s) was damped to 4.8 m/s!


