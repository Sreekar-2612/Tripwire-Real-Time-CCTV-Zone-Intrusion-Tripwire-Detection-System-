import pytest
import numpy as np

from cv_core.kinetics.pose_classifier import PoseKineticClassifier, PostureType

@pytest.fixture
def classifier():
    return PoseKineticClassifier(keypoint_conf_threshold=0.3)

def test_climbing_classification(classifier):
    """Verify that raised hands near perimeter triggers CLIMBING threat."""
    # Construct 17 keypoints for an upright person with hands held high above head
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.95  # high confidence

    # Head and shoulders
    kpts[PoseKineticClassifier.NOSE] = [100.0, 100.0, 0.95]
    kpts[PoseKineticClassifier.LEFT_SHOULDER] = [80.0, 130.0, 0.95]
    kpts[PoseKineticClassifier.RIGHT_SHOULDER] = [120.0, 130.0, 0.95]
    # Wrists raised high above shoulders (lower Y in image coords)
    kpts[PoseKineticClassifier.LEFT_WRIST] = [70.0, 70.0, 0.95]
    kpts[PoseKineticClassifier.RIGHT_WRIST] = [130.0, 70.0, 0.95]
    # Hips and legs
    kpts[PoseKineticClassifier.LEFT_HIP] = [85.0, 250.0, 0.95]
    kpts[PoseKineticClassifier.RIGHT_HIP] = [115.0, 250.0, 0.95]

    res = classifier.classify_pose(kpts, speed_mps=0.3, is_near_perimeter=True)
    assert res["posture"] == PostureType.CLIMBING.value
    assert res["threat_level"] >= 8

def test_crawling_classification(classifier):
    """Verify that horizontal torso with locomotion triggers CRAWLING."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9

    # Prone torso (horizontal: shoulders and hips at approximately same Y level)
    kpts[PoseKineticClassifier.LEFT_SHOULDER] = [200.0, 400.0, 0.9]
    kpts[PoseKineticClassifier.RIGHT_SHOULDER] = [200.0, 410.0, 0.9]
    kpts[PoseKineticClassifier.LEFT_HIP] = [100.0, 405.0, 0.9]
    kpts[PoseKineticClassifier.RIGHT_HIP] = [100.0, 415.0, 0.9]

    # Moving at 0.5 m/s crawling speed
    res = classifier.classify_pose(kpts, speed_mps=0.5, is_near_perimeter=False)
    assert res["posture"] == PostureType.CRAWLING.value
    assert res["threat_level"] >= 7

def test_fallen_person_classification(classifier):
    """Verify that horizontal stationary posture triggers FALLEN."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9

    # Prone torso, stationary speed = 0.0
    kpts[PoseKineticClassifier.LEFT_SHOULDER] = [200.0, 400.0, 0.9]
    kpts[PoseKineticClassifier.RIGHT_SHOULDER] = [200.0, 410.0, 0.9]
    kpts[PoseKineticClassifier.LEFT_HIP] = [100.0, 405.0, 0.9]
    kpts[PoseKineticClassifier.RIGHT_HIP] = [100.0, 415.0, 0.9]

    res = classifier.classify_pose(kpts, speed_mps=0.0)
    assert res["posture"] == PostureType.FALLEN.value

def test_running_vs_walking_speed_gating(classifier):
    """Verify running classification when speed >= 3.0 m/s."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9
    kpts[PoseKineticClassifier.LEFT_SHOULDER] = [100.0, 100.0, 0.9]
    kpts[PoseKineticClassifier.RIGHT_SHOULDER] = [120.0, 100.0, 0.9]
    kpts[PoseKineticClassifier.LEFT_HIP] = [100.0, 200.0, 0.9]
    kpts[PoseKineticClassifier.RIGHT_HIP] = [120.0, 200.0, 0.9]

    # Speed 3.5 m/s = 12.6 km/h -> RUNNING
    run_res = classifier.classify_pose(kpts, speed_mps=3.5)
    assert run_res["posture"] == PostureType.RUNNING.value

    # Speed 1.2 m/s -> WALKING
    walk_res = classifier.classify_pose(kpts, speed_mps=1.2)
    assert walk_res["posture"] == PostureType.WALKING.value

def test_insufficient_keypoints_graceful_handling(classifier):
    """Verify that occluded keypoints return UNKNOWN without crashing."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.0  # zero confidence for all joints

    res = classifier.classify_pose(kpts, speed_mps=1.0)
    assert res["posture"] == PostureType.UNKNOWN.value
    assert res["confidence"] == 0.0
