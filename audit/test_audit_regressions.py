"""Audit probes assert desired behavior; failures document unfixed defects."""
import numpy as np
from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D
from cv_core.tracking.tracker import SingleTrack, MultiTargetTracker
from cv_core.kinetics.pose_classifier import PoseKineticClassifier
from evaluation.evaluate_mot import MOTEvaluator


def test_tripwire_crossing_with_sample_on_line():
    zone = MetricZone3D('t', 'test', 'tripwire', [(0, 0), (10, 0)])
    from cv_core.monitoring import ZoneMonitor
    monitor = ZoneMonitor()
    events = []
    for point in ((5, 1), (5, 0), (5, -1)):
        _, emitted = monitor.update([{'track_id': 1, 'world_pos': point}], [zone], .1)
        events.extend(emitted)
    assert [e['event'] for e in events] == ['CROSS_IN']



def test_recovery_speed_accounts_for_missing_frames():
    tracker = MultiTargetTracker(min_hits=1, dt=0.04, ema_alpha=1.0, iou_threshold=0.1)
    h = HomographyTransformer(np.diag([0.01, 0.01, 1.0]))
    def detection(x):
        return [{'bbox': [x, 0, x + 100, 200], 'confidence': .9}]
    for x in [0, 4, 8]:
        tracker.update(detection(x), h)
    for _ in range(4):
        tracker.update([], h)
    out = tracker.update(detection(28), h)
    assert len(out) == 1
    assert np.isclose(out[0]['speed'], 3.6, atol=.1), out[0]['speed']


def test_climbing_is_reachable_through_tracker():
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 2] = .99
    k[5] = [100, 100, .99]; k[6] = [120, 100, .99]
    k[11] = [100, 170, .99]; k[12] = [120, 170, .99]
    k[9] = [100, 50, .99]; k[10] = [120, 50, .99]
    k[15] = [100, 200, .99]; k[16] = [120, 200, .99]
    classifier = PoseKineticClassifier()
    assert classifier.classify_pose(k, is_near_perimeter=True)['posture'] == 'climbing'
    tracker = MultiTargetTracker(min_hits=1)
    det = [{'bbox': [80, 40, 140, 200], 'confidence': .99, 'keypoints': k}]
    wire = MetricZone3D('w', 'Perimeter', 'tripwire', [(80, 200), (140, 200)])
    for _ in range(8):
        out = tracker.update(det, HomographyTransformer(np.eye(3)), classifier, zones=[wire])
    assert out[0]['posture'] == 'climbing', out[0]['posture']


def test_nan_calibration_rejected():
    import pytest
    with pytest.raises(ValueError):
        HomographyTransformer(np.full((3, 3), np.nan))


def test_reprojection_all_invalid_is_not_perfect():
    h = HomographyTransformer(np.eye(3))
    assert h.compute_reprojection_error([(np.nan, 0)], [(0, 0)]) != 0


def test_mot_matching_does_not_depend_on_gt_order():
    e = MOTEvaluator(.5)
    a = {'track_id': 1, 'bbox': [0, 0, 10, 10]}
    b = {'track_id': 2, 'bbox': [4, 0, 14, 10]}
    p = {1: [{'track_id': 101, 'bbox': [1, 0, 11, 10]},
             {'track_id': 102, 'bbox': [-2, 0, 8, 10]}]}
    forward = e.evaluate_sequence({1: [a, b]}, p)
    reverse = e.evaluate_sequence({1: [b, a]}, p)
    assert forward['true_positives'] == reverse['true_positives'], (forward, reverse)


def test_tentative_track_cannot_confirm_after_one_miss():
    tracker = MultiTargetTracker(min_hits=3)
    d = [{'bbox': [0, 0, 10, 20], 'confidence': .9}]
    assert not tracker.update(d)
    tracker.update([])
    assert not tracker.update(d)
