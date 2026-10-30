import pytest
import numpy as np

from cv_core.tracking.kalman_filter import KalmanFilter2D
from cv_core.tracking.tracker import MultiTargetTracker, compute_iou_matrix, TrackState

def test_kalman_filter_prediction_and_velocity_convergence():
    """Verify that Kalman filter accurately tracks linear motion and converges velocity."""
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
