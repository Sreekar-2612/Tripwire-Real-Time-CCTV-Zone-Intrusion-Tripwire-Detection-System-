from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

class PostureType(Enum):
    STANDING = "standing"
    WALKING = "walking"
    RUNNING = "running"
    CRAWLING = "crawling"
    CLIMBING = "climbing"
    FALLEN = "fallen"
    UNKNOWN = "unknown"

class PoseKineticClassifier:
    """
    Kinetic & Biomechanical Action Recognition Engine.
    Analyzes 17 COCO human pose keypoints to detect security threats:
    - Fence/wall climbing
    - Low-profile crawling under tripwires
    - Rapid running/fleeing
    - Person fallen / medical emergency
    """

    # COCO Keypoint Index Mapping
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(self, keypoint_conf_threshold: float = 0.35):
        self.conf_thresh = keypoint_conf_threshold

    def classify_pose(
        self,
        keypoints: np.ndarray,
        speed_mps: float = 0.0,
        is_near_perimeter: bool = False
    ) -> Dict[str, Any]:
        """
        keypoints: (17, 2) or (17, 3) numpy array [x, y] or [x, y, conf].
        speed_mps: Optional real-world speed in m/s from homography tracker.
        
        Returns:
            posture: PostureType string
            threat_level: int [1, 10]
            confidence: float [0.0, 1.0]
            indicators: List of descriptive string tags
        """
        kpts = np.asarray(keypoints, dtype=np.float32)
        has_conf = kpts.shape[1] >= 3

        def get_pt(idx: int) -> Optional[Tuple[float, float]]:
            if has_conf and kpts[idx, 2] < self.conf_thresh:
                return None
            return (float(kpts[idx, 0]), float(kpts[idx, 1]))

        # Extract major joints
        nose = get_pt(self.NOSE)
        l_sh = get_pt(self.LEFT_SHOULDER)
        r_sh = get_pt(self.RIGHT_SHOULDER)
        l_hip = get_pt(self.LEFT_HIP)
        r_hip = get_pt(self.RIGHT_HIP)
        l_wrist = get_pt(self.LEFT_WRIST)
        r_wrist = get_pt(self.RIGHT_WRIST)
        l_ankle = get_pt(self.LEFT_ANKLE)
        r_ankle = get_pt(self.RIGHT_ANKLE)

        indicators = []

        # 1. Compute Midpoints
        mid_shoulder = None
        if l_sh and r_sh:
            mid_shoulder = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        elif l_sh:
            mid_shoulder = l_sh
        elif r_sh:
            mid_shoulder = r_sh

        mid_hip = None
        if l_hip and r_hip:
            mid_hip = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
        elif l_hip:
            mid_hip = l_hip
        elif r_hip:
            mid_hip = r_hip

        if mid_shoulder is None or mid_hip is None:
            # Insufficient keypoint coverage
            return {
                "posture": PostureType.UNKNOWN.value,
                "threat_level": 1,
                "confidence": 0.0,
                "indicators": ["insufficient_keypoint_confidence"]
            }

        # Vector from Hip to Shoulder
        torso_dx = mid_shoulder[0] - mid_hip[0]
        torso_dy = mid_shoulder[1] - mid_hip[1]  # In image coordinates, Y points DOWN (ground)

        # Torso inclination angle relative to ground plane (0 = horizontal, 90 = vertical upright)
        torso_angle_deg = abs(np.degrees(np.arctan2(abs(torso_dy), abs(torso_dx) + 1e-6)))

        # 2. Test CLIMBING: Hands above head/shoulders while body is extended near perimeter
        hands_elevated = False
        if mid_shoulder is not None:
            # In image coords, above means lower Y
            if l_wrist and l_wrist[1] < (mid_shoulder[1] - 15.0):
                hands_elevated = True
            if r_wrist and r_wrist[1] < (mid_shoulder[1] - 15.0):
                hands_elevated = True

        if hands_elevated and (is_near_perimeter or torso_angle_deg > 60.0):
            indicators.append("hands_elevated_above_shoulders")
            if is_near_perimeter:
                indicators.append("perimeter_proximity_scaling")
                return {
                    "posture": PostureType.CLIMBING.value,
                    "threat_level": 9,
                    "confidence": 0.88,
                    "indicators": indicators
                }

        # 3. Test CRAWLING / PRONE (Horizontal torso, hips and shoulders close to ground)
        if torso_angle_deg < 35.0:
            indicators.append(f"horizontal_torso_angle_{torso_angle_deg:.1f}deg")
            if speed_mps > 0.2:
                indicators.append("prone_locomotion")
                return {
                    "posture": PostureType.CRAWLING.value,
                    "threat_level": 8,
                    "confidence": 0.85,
                    "indicators": indicators
                }
            else:
                indicators.append("prone_stationary")
                return {
                    "posture": PostureType.FALLEN.value,
                    "threat_level": 6,
                    "confidence": 0.80,
                    "indicators": indicators
                }

        # 4. Test RUNNING vs WALKING vs STANDING via Speed & Stride Extension
        stride_span = 0.0
        if l_ankle and r_ankle:
            stride_span = abs(l_ankle[0] - r_ankle[0])

        if speed_mps >= 3.0:
            indicators.append(f"high_velocity_{speed_mps:.1f}mps")
            return {
                "posture": PostureType.RUNNING.value,
                "threat_level": 6,
                "confidence": 0.90,
                "indicators": indicators
            }
        elif speed_mps >= 0.8:
            indicators.append(f"moderate_velocity_{speed_mps:.1f}mps")
            return {
                "posture": PostureType.WALKING.value,
                "threat_level": 2,
                "confidence": 0.85,
                "indicators": indicators
            }
        else:
            return {
                "posture": PostureType.STANDING.value,
                "threat_level": 1,
                "confidence": 0.80,
                "indicators": ["upright_stationary"]
            }
