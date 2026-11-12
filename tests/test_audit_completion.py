import json
from unittest.mock import Mock

import numpy as np
import pytest

from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D
from cv_core.tracking.tracker import MultiTargetTracker


def test_explicit_frame_intervals_accumulate_across_occlusion():
    tracker = MultiTargetTracker(min_hits=1, dt=.04, ema_alpha=1, iou_threshold=.1)
    h = HomographyTransformer(np.diag([.01, .01, 1]))
    def step(x, dt):
        det = [] if x is None else [{'bbox': [x, 0, x+100, 200], 'confidence': .9}]
        return tracker.update(det, h, elapsed_dt=dt)
    for x in (0, 4, 8):
        step(x, .04)
    step(None, .08)
    step(None, .12)
    out = step(32, .04)
    assert out[0]['speed'] == pytest.approx(3.6)
    assert tracker.tracks[0].kf.dt == pytest.approx(.04)


def test_wire_touch_is_not_crossing_and_crossing_respects_segment():
    from cv_core.monitoring import ZoneMonitor
    wire = MetricZone3D('w', 'Wire', 'tripwire', [(0, 0), (10, 0)])
    def run(points):
        m = ZoneMonitor()
        events = []
        for point in points:
            _, emitted = m.update([{'track_id': 1, 'world_pos': point}], [wire], .1)
            events.extend(e['event'] for e in emitted)
        return events
    assert run([(5, 1), (5, 0), (5, 1)]) == []
    assert run([(5, 1), (5, 0), (5, -1)]) == ['CROSS_IN']
    assert run([(15, 1), (15, 0), (15, -1)]) == []


def test_incidents_do_not_repeat_or_starve_other_people():
    from cv_core.monitoring import ZoneMonitor
    zone = MetricZone3D('p', 'Vault', 'polygon', [(0, 0), (10, 0), (10, 10), (0, 10)])
    wire = MetricZone3D('w', 'Wire', 'tripwire', [(0, 5), (10, 5)])
    m = ZoneMonitor()
    def tracks(y):
        return [{'track_id': n, 'world_pos': (n, y)} for n in (1, 2)]
    _, first = m.update(tracks(6), [zone, wire], .5, dwell_confirm_sec=1)
    state, second = m.update(tracks(4), [zone, wire], .5, dwell_confirm_sec=1)
    assert [e['event'] for e in first] == ['ENTER', 'ENTER']
    assert sum(e['event'] == 'CROSS_IN' for e in second) == 2
    assert all(s['is_breaching'] for s in state.values())
    _, third = m.update(tracks(4), [zone, wire], .5, dwell_confirm_sec=1)
    assert sum(e['event'] == 'LOITER' for e in third) == 2
    _, fourth = m.update(tracks(4), [zone, wire], .5, dwell_confirm_sec=1)
    assert not fourth
    _, last = m.update([], [zone, wire], .5, dwell_confirm_sec=1)
    assert sum(e['event'] == 'EXIT' for e in last) == 2
    assert not m.inside and not m.wire_points


def test_calibration_is_bound_to_source_and_frame_size(tmp_path):
    h = HomographyTransformer(np.eye(3))
    calibration = tmp_path / 'calibration.json'
    h.save_calibration(calibration, source='camera-a.mp4', image_size=(640, 480))
    assert HomographyTransformer.for_source('camera-a.mp4', (640, 480), calibration).is_calibrated
    with pytest.raises(ValueError, match='source'):
        HomographyTransformer.for_source('camera-b.mp4', (640, 480), calibration)
    with pytest.raises(ValueError, match='size'):
        HomographyTransformer.for_source('camera-a.mp4', (320, 240), calibration)
    assert not HomographyTransformer.for_source('unconfigured.mp4', (640, 480)).is_calibrated
    data = json.loads(calibration.read_text())
    data.pop('image_size')
    calibration.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='together'):
        HomographyTransformer.for_source('camera-a.mp4', (640, 480), calibration)


def test_reprojection_rejects_mismatched_points_and_invalid_bounds():
    h = HomographyTransformer(np.eye(3))
    with pytest.raises(ValueError):
        h.compute_reprojection_error([(0, 0), (1, 1)], [(0, 0)])
    with pytest.raises(ValueError):
        HomographyTransformer(np.eye(3), world_bounds=(0, 0, 0, np.nan))


def test_polygon_proximity_uses_current_point_without_a_tripwire():
    from cv_core.kinetics.pose_classifier import PoseKineticClassifier
    classifier = Mock(wraps=PoseKineticClassifier())
    tracker = MultiTargetTracker(min_hits=1)
    zone = MetricZone3D('p', 'Vault', 'polygon', [(0, 0), (10, 0), (10, 10), (0, 10)])
    keypoints = np.zeros((17, 3))
    det = [{'bbox': [0, 0, 10, 10], 'confidence': .9, 'keypoints': keypoints}]
    for _ in range(2):
        tracker.update(det, HomographyTransformer(np.eye(3)), classifier, zones=[zone])
    assert classifier.classify_pose.call_args.kwargs['is_near_perimeter'] is True


def test_file_clock_uses_source_time_and_live_clock_uses_elapsed_time():
    import cv2
    from cv_core.video import FrameClock
    cap = Mock()
    cap.get.side_effect = lambda key: 12.0 if key == cv2.CAP_PROP_FPS else 0.0
    file_clock = FrameClock(cap, live=False)
    assert file_clock.tick() == pytest.approx(1/12)
    ticks = iter([10, 10.2, 10.7])
    live_clock = FrameClock(cap, live=True, now=lambda: next(ticks))
    assert live_clock.tick() == pytest.approx(.2)
    assert live_clock.tick() == pytest.approx(.5)


def test_mot_assignment_maximizes_valid_matches_before_overlap():
    from evaluation.evaluate_mot import MOTEvaluator
    evaluator = MOTEvaluator(.5)
    overlaps = [[.9, .51], [.51, .49]]
    evaluator.calculate_box_iou = lambda a, b: overlaps[a[0]][b[0]]
    gt = {1: [{'track_id': i, 'bbox': [i]} for i in range(2)]}
    predictions = {1: [{'track_id': i+10, 'bbox': [i]} for i in range(2)]}
    assert evaluator.evaluate_sequence(gt, predictions)['true_positives'] == 2
