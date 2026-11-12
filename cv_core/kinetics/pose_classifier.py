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

# Standard COCO 17 Keypoint Skeleton Connectivity
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # Head & Face
    (5, 6),                                # Shoulders
    (5, 7), (7, 9),                        # Left Arm
    (6, 8), (8, 10),                       # Right Arm
    (5, 11), (6, 12), (11, 12),            # Torso
    (11, 13), (13, 15),                    # Left Leg
    (12, 14), (14, 16)                     # Right Leg
]

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
    LEFT_EAR = 3
    RIGHT_EAR = 4
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
        keypoints: Any,
        speed_mps: float = 0.0,
        is_near_perimeter: bool = False
    ) -> Dict[str, Any]:
        """
        Classifies human posture from 17 keypoint coordinates.

        keypoints: (17, 2) or (17, 3) array [x, y] or [x, y, conf], or PyTorch Tensor.
        speed_mps: Optional real-world speed in m/s from homography tracker.
        is_near_perimeter: Whether target is adjacent to a tripwire or restricted boundary.
        
        Returns:
            posture: PostureType string ('climbing', 'crawling', 'fallen', 'running', 'walking', 'standing', 'unknown')
            threat_level: int [1, 10]
            confidence: float [0.0, 1.0]
            indicators: List of descriptive string tags
        """
        if keypoints is None:
            return {
                "posture": PostureType.UNKNOWN.value,
                "threat_level": 1,
                "confidence": 0.0,
                "indicators": ["missing_keypoints"]
            }

        if hasattr(keypoints, "cpu") and hasattr(keypoints, "numpy"):
            kpts = keypoints.cpu().numpy()
        else:
            try:
                kpts = np.asarray(keypoints, dtype=np.float32)
            except Exception:
                return {
                    "posture": PostureType.UNKNOWN.value,
                    "threat_level": 1,
                    "confidence": 0.0,
                    "indicators": ["invalid_keypoints_format"]
                }

        if kpts.ndim == 3 and kpts.shape[0] == 1:
            kpts = kpts[0]

        if kpts.ndim != 2 or kpts.shape[0] < 17 or kpts.shape[1] < 2:
            return {
                "posture": PostureType.UNKNOWN.value,
                "threat_level": 1,
                "confidence": 0.0,
                "indicators": ["invalid_keypoint_count"]
            }

        speed_mps = float(speed_mps) if (speed_mps is not None and np.isfinite(speed_mps)) else 0.0
        has_conf = kpts.shape[1] >= 3

        def get_pt(idx: int) -> Optional[Tuple[float, float]]:
            if idx >= len(kpts):
                return None
            if has_conf and (not np.isfinite(kpts[idx, 2]) or kpts[idx, 2] < self.conf_thresh):
                return None
            x, y = float(kpts[idx, 0]), float(kpts[idx, 1])
            if not (np.isfinite(x) and np.isfinite(y)):
                return None
            return (x, y)

        # Extract major joints
        nose = get_pt(self.NOSE)
        l_sh = get_pt(self.LEFT_SHOULDER)
        r_sh = get_pt(self.RIGHT_SHOULDER)
        l_hip = get_pt(self.LEFT_HIP)
        r_hip = get_pt(self.RIGHT_HIP)
        l_wrist = get_pt(self.LEFT_WRIST)
        r_wrist = get_pt(self.RIGHT_WRIST)
        l_knee = get_pt(self.LEFT_KNEE)
        r_knee = get_pt(self.RIGHT_KNEE)
        l_ankle = get_pt(self.LEFT_ANKLE)
        r_ankle = get_pt(self.RIGHT_ANKLE)

        indicators: List[str] = []

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

        if hands_elevated and is_near_perimeter:
            indicators.append("hands_elevated_above_shoulders")
            indicators.append("perimeter_proximity_scaling")
            return {
                "posture": PostureType.CLIMBING.value,
                "threat_level": 9,
                "confidence": 0.88,
                "indicators": indicators
            }
        elif hands_elevated:
            indicators.append("hands_elevated_above_shoulders")

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

        # 4. Stride Span & Leg Dynamics
        stride_span = 0.0
        if l_ankle and r_ankle:
            stride_span = abs(l_ankle[0] - r_ankle[0])

        body_height = max(abs(mid_shoulder[1] - (l_ankle[1] if l_ankle else mid_hip[1])), 1.0)
        stride_ratio = stride_span / body_height

        # 5. Test RUNNING vs WALKING vs STANDING via Speed & Stride Extension
        if speed_mps >= 3.0 or stride_ratio > 0.85:
            indicators.append(f"high_velocity_{speed_mps:.1f}mps" if speed_mps >= 3.0 else f"extended_stride_ratio_{stride_ratio:.2f}")
            return {
                "posture": PostureType.RUNNING.value,
                "threat_level": 6,
                "confidence": 0.90,
                "indicators": indicators
            }
        elif speed_mps >= 0.8 or stride_ratio > 0.40:
            indicators.append(f"moderate_velocity_{speed_mps:.1f}mps" if speed_mps >= 0.8 else f"walking_stride_ratio_{stride_ratio:.2f}")
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

def classify_keypoints_posture(
    keypoints: Any,
    speed_mps: float = 0.0,
    is_near_perimeter: bool = False,
    keypoint_conf_threshold: float = 0.35
) -> Dict[str, Any]:
    """
    Direct helper to classify human posture from YOLO-Pose keypoints.
    Supports PyTorch Tensors, NumPy arrays, or coordinate lists.
    """
    classifier = PoseKineticClassifier(keypoint_conf_threshold=keypoint_conf_threshold)
    return classifier.classify_pose(keypoints, speed_mps=speed_mps, is_near_perimeter=is_near_perimeter)

def extract_and_classify_yolo_pose(
    yolo_result: Any,
    speed_mps_list: Optional[List[float]] = None,
    perimeter_flags: Optional[List[bool]] = None,
    keypoint_conf_threshold: float = 0.35
) -> List[Dict[str, Any]]:
    """
    Extracts keypoints from an Ultralytics YOLO-Pose result object and classifies each target.
    """
    if yolo_result is None or not hasattr(yolo_result, "keypoints") or yolo_result.keypoints is None:
        return []

    kpts_data = yolo_result.keypoints.data
    if kpts_data is None or len(kpts_data) == 0:
        return []

    if hasattr(kpts_data, "cpu"):
        kpts_np = kpts_data.cpu().numpy()
    else:
        kpts_np = np.asarray(kpts_data, dtype=np.float32)

    classifier = PoseKineticClassifier(keypoint_conf_threshold=keypoint_conf_threshold)
    results = []
    num_persons = kpts_np.shape[0]

    for i in range(num_persons):
        speed = speed_mps_list[i] if speed_mps_list and i < len(speed_mps_list) else 0.0
        near_peri = perimeter_flags[i] if perimeter_flags and i < len(perimeter_flags) else False
        res = classifier.classify_pose(kpts_np[i], speed_mps=speed, is_near_perimeter=near_peri)
        results.append(res)

    return results

def draw_pose_skeleton(
    frame: np.ndarray,
    keypoints: Any,
    conf_thresh: float = 0.35,
    joint_color: Tuple[int, int, int] = (0, 255, 200),
    limb_color: Tuple[int, int, int] = (255, 180, 0)
) -> np.ndarray:
    """
    Visualizes 17 COCO keypoints and connecting skeleton limbs on an OpenCV image frame.
    """
    import cv2

    if hasattr(keypoints, "cpu") and hasattr(keypoints, "numpy"):
        kpts = keypoints.cpu().numpy()
    else:
        kpts = np.asarray(keypoints, dtype=np.float32)

    if kpts.ndim == 3 and kpts.shape[0] == 1:
        kpts = kpts[0]
    elif kpts.ndim == 1:
        if kpts.shape[0] == 51:
            kpts = kpts.reshape((17, 3))
        elif kpts.shape[0] == 34:
            kpts = kpts.reshape((17, 2))
        else:
            return frame

    if kpts.ndim != 2 or kpts.shape[0] < 17 or kpts.shape[1] < 2:
        return frame

    has_conf = kpts.shape[1] >= 3

    # Draw skeleton limbs
    for idx1, idx2 in COCO_SKELETON:
        if has_conf and (kpts[idx1, 2] < conf_thresh or kpts[idx2, 2] < conf_thresh):
            continue
        if not (np.isfinite(kpts[idx1, 0]) and np.isfinite(kpts[idx1, 1]) and np.isfinite(kpts[idx2, 0]) and np.isfinite(kpts[idx2, 1])):
            continue
        p1 = (int(kpts[idx1, 0]), int(kpts[idx1, 1]))
        p2 = (int(kpts[idx2, 0]), int(kpts[idx2, 1]))
        cv2.line(frame, p1, p2, limb_color, 2, cv2.LINE_AA)

    # Draw joint nodes
    for i in range(17):
        if has_conf and kpts[i, 2] < conf_thresh:
            continue
        if not (np.isfinite(kpts[i, 0]) and np.isfinite(kpts[i, 1])):
            continue
        pt = (int(kpts[i, 0]), int(kpts[i, 1]))
        cv2.circle(frame, pt, 3, joint_color, -1, cv2.LINE_AA)

    return frame
