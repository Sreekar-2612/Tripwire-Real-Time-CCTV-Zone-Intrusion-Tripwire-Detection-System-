import pytest
import numpy as np
import cv2

from cv_core.tracking.tracker import (
    MultiTargetTracker,
    SingleTrack,
    compute_iou_matrix,
    compute_diou_matrix,
    linear_assignment_iou
)
from cv_core.geometry.homography import HomographyTransformer
from cv_core.kinetics.pose_classifier import PoseKineticClassifier, PostureType
from cv_core.visualizer.bev_radar import BEVRadarRenderer
from cv_core.geometry.zones_3d import MetricZone3D


def test_compute_diou_matrix_properties():
    """Verify mathematical properties of DIoU matrix calculation."""
    # 1. Exact match -> IoU = 1.0, distance = 0 -> DIoU = 1.0
    boxes_a = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
    boxes_b = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
    diou = compute_diou_matrix(boxes_a, boxes_b)
    assert np.isclose(diou[0, 0], 1.0, atol=1e-5)

    # 2. Concentric boxes of different size -> center distance = 0 -> DIoU == IoU
    boxes_c = np.array([[20.0, 20.0, 40.0, 40.0]], dtype=np.float32)
    iou = compute_iou_matrix(boxes_a, boxes_c)
    diou_concentric = compute_diou_matrix(boxes_a, boxes_c)
    assert np.isclose(diou_concentric[0, 0], iou[0, 0], atol=1e-5)

    # 3. Disjoint non-overlapping boxes -> IoU = 0.0, DIoU < 0.0 and >= -1.0
    boxes_disjoint = np.array([[100.0, 100.0, 140.0, 140.0]], dtype=np.float32)
    diou_disjoint = compute_diou_matrix(boxes_a, boxes_disjoint)
    assert diou_disjoint[0, 0] < 0.0
    assert diou_disjoint[0, 0] >= -1.0

    # 4. DIoU is always <= IoU
    boxes_rand_a = np.random.uniform(10, 100, (5, 4)).astype(np.float32)
    boxes_rand_a[:, 2:] += boxes_rand_a[:, :2]  # ensure x2 > x1, y2 > y1
    boxes_rand_b = np.random.uniform(10, 100, (5, 4)).astype(np.float32)
    boxes_rand_b[:, 2:] += boxes_rand_b[:, :2]

    mat_iou = compute_iou_matrix(boxes_rand_a, boxes_rand_b)
    mat_diou = compute_diou_matrix(boxes_rand_a, boxes_rand_b)
    assert np.all(mat_diou <= mat_iou + 1e-6)


def test_diou_matching_resolves_fast_moving_targets():
    """Verify that Hungarian assignment with DIoU matches fast targets with zero IoU overlap."""
    SingleTrack.reset_id_counter(10)
    tracker = MultiTargetTracker(min_hits=1, max_age=5, iou_threshold=0.0)

    # Initial frame
    det1 = [{"bbox": [50.0, 50.0, 90.0, 130.0], "confidence": 0.9, "foot_point": (70.0, 130.0)}]
    out1 = tracker.update(det1)
    orig_id = out1[0]["track_id"]

    # Target makes high-speed jump of 45 pixels (box width is 40px, so IoU = 0)
    det2 = [{"bbox": [95.0, 50.0, 135.0, 130.0], "confidence": 0.88, "foot_point": (115.0, 130.0)}]
    out2 = tracker.update(det2)

    assert len(out2) == 1
    assert out2[0]["track_id"] == orig_id  # Track ID preserved via DIoU proximity!


def test_ankle_keypoints_ground_contact_refinement():
    """Verify that ankle keypoints (COCO joints 15 & 16) refine foot contact on ground."""
    tracker = MultiTargetTracker(min_hits=1, max_age=5)

    # 17 keypoints array with left ankle (15) and right ankle (16)
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.95
    # Bbox bottom center would be ((100+140)/2, 200) = (120.0, 200.0)
    # Ankles are at (118.0, 196.0) and (124.0, 194.0) -> midpoint (121.0, 195.0)
    kpts[15] = [118.0, 196.0, 0.95]
    kpts[16] = [124.0, 194.0, 0.95]

    det = {
        "bbox": [100.0, 100.0, 140.0, 200.0],
        "confidence": 0.92,
        "foot_point": (120.0, 200.0),
        "keypoints": kpts
    }

    out = tracker.update([det])
    assert len(out) == 1
    # Foot point must match ankle midpoint (121.0, 195.0), not bbox bottom (120.0, 200.0)
    fx, fy = out[0]["foot_point"]
    assert np.isclose(fx, 121.0, atol=0.5)
    assert np.isclose(fy, 195.0, atol=0.5)


def test_temporal_posture_smoothing_suppresses_transient_glitch():
    """Verify temporal smoothing filters out single-frame posture drops while preserving real threats."""
    trk = SingleTrack(bbox=[50.0, 50.0, 90.0, 130.0], confidence=0.9, foot_point=(70.0, 130.0))

    # Frame 1 & 2: Running
    trk.update([50.0, 50.0, 90.0, 130.0], 0.9, (70.0, 130.0), posture_info={"posture": "running", "threat_level": 6, "indicators": []})
    trk.update([55.0, 50.0, 95.0, 130.0], 0.9, (75.0, 130.0), posture_info={"posture": "running", "threat_level": 6, "indicators": []})
    assert trk.posture == "running"

    # Frame 3: Single-frame noisy keypoint misclassifies as STANDING
    trk.update([60.0, 50.0, 100.0, 130.0], 0.9, (80.0, 130.0), posture_info={"posture": "standing", "threat_level": 1, "indicators": []})
    # Must remain running due to majority history smoothing!
    assert trk.posture == "running"

    # Frame 4 & 5: Threat posture CRAWLING confirmed in 2 frames
    trk.update([65.0, 50.0, 105.0, 130.0], 0.9, (85.0, 130.0), posture_info={"posture": "crawling", "threat_level": 8, "indicators": []})
    trk.update([70.0, 50.0, 110.0, 130.0], 0.9, (90.0, 130.0), posture_info={"posture": "crawling", "threat_level": 8, "indicators": []})
    # Immediate escalation to crawling threat!
    assert trk.posture == "crawling"


def test_bev_radar_static_background_caching_integrity():
    """Verify that BEV radar static background cache is valid and identical to fresh renders."""
    renderer = BEVRadarRenderer(radar_size=400, world_extent_m=20.0, grid_step_m=2.0)
    assert hasattr(renderer, "_static_bg")
    assert renderer._static_bg.shape == (400, 400, 3)

    # Calling render multiple times produces consistent output
    zones = [MetricZone3D("z1", "TestZone", "polygon", [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)])]
    tracks = [{"track_id": 1, "world_pos": (4.0, 4.0), "world_trail": [(4.0, 4.0)], "speed_kmh": 0.0, "vx": 0.0, "vy": 0.0, "is_breaching": False}]
    
    r1 = renderer.render(zones, tracks)
    r2 = renderer.render(zones, tracks)
    assert np.array_equal(r1, r2)
