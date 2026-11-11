from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, TYPE_CHECKING
import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from cv_core.geometry.homography import HomographyTransformer

from cv_core.tracking.kalman_filter import KalmanFilter2D
from cv_core.kinetics.pose_classifier import (
    PostureType,
    PoseKineticClassifier,
    classify_keypoints_posture,
    draw_pose_skeleton
)

class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"

def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes Intersection-over-Union (IoU) matrix between two sets of [x1, y1, x2, y2] boxes.
    Returns (N, M) matrix with IoU values in [0.0, 1.0].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    if not np.all(np.isfinite(boxes_a)) or not np.all(np.isfinite(boxes_b)):
        boxes_a = np.nan_to_num(boxes_a, nan=0.0, posinf=1e5, neginf=0.0)
        boxes_b = np.nan_to_num(boxes_b, nan=0.0, posinf=1e5, neginf=0.0)

    area_a = np.maximum(0.0, (boxes_a[:, 2] - boxes_a[:, 0])) * np.maximum(0.0, (boxes_a[:, 3] - boxes_a[:, 1]))
    area_b = np.maximum(0.0, (boxes_b[:, 2] - boxes_b[:, 0])) * np.maximum(0.0, (boxes_b[:, 3] - boxes_b[:, 1]))

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    inter_wh = np.maximum(0.0, bottom_right - top_left)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    union_area = np.maximum(union_area, 1e-6)

    iou = inter_area / union_area
    return np.nan_to_num(iou, nan=0.0)

def compute_diou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes Distance-IoU (DIoU) matrix between two sets of [x1, y1, x2, y2] boxes.
    DIoU = IoU - (rho^2 / c^2)
    where rho is Euclidean distance between box centers, and c is diagonal of the smallest enclosing box.
    Returns (N, M) matrix with DIoU values in [-1.0, 1.0].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    iou = compute_iou_matrix(boxes_a, boxes_b)

    # Centers of boxes
    cx_a = (boxes_a[:, 0] + boxes_a[:, 2]) / 2.0
    cy_a = (boxes_a[:, 1] + boxes_a[:, 3]) / 2.0
    cx_b = (boxes_b[:, 0] + boxes_b[:, 2]) / 2.0
    cy_b = (boxes_b[:, 1] + boxes_b[:, 3]) / 2.0

    # Euclidean distance between centers: rho^2
    d_center_sq = (cx_a[:, None] - cx_b[None, :]) ** 2 + (cy_a[:, None] - cy_b[None, :]) ** 2

    # Smallest enclosing box
    enc_x1 = np.minimum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    enc_y1 = np.minimum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    enc_x2 = np.maximum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    enc_y2 = np.maximum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    c_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
    c_sq = np.maximum(c_sq, 1e-6)

    diou = iou - (d_center_sq / c_sq)
    return np.nan_to_num(diou, nan=-1.0)

def linear_assignment_iou(
    tracks: List["SingleTrack"],
    detections: List[Dict[str, Any]],
    iou_thresh: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solves optimal bipartite matching between tracks and detections using Hungarian assignment.
    Combines DIoU distance affinity with IoU acceptance gating.
    Returns:
        matches: List of (track_idx, det_idx) pairs satisfying IoU >= iou_thresh
        unmatched_tracks: List of indices of tracks with no valid match
        unmatched_dets: List of indices of detections with no valid match
    """
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    track_boxes = np.array([t.bbox for t in tracks], dtype=np.float32)
    det_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)

    iou_matrix = compute_iou_matrix(track_boxes, det_boxes)
    diou_matrix = compute_diou_matrix(track_boxes, det_boxes)
    # DIoU cost matrix in [0, 2]: minimizes both bounding box overlap loss and center distance
    cost_matrix = 1.0 - diou_matrix
    cost_matrix = np.nan_to_num(cost_matrix, nan=2.0, posinf=2.0, neginf=2.0)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_thresh:
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

    return matches, sorted(list(unmatched_tracks)), sorted(list(unmatched_dets))

class SingleTrack:
    """
    Represents a single tracked target with state estimation, lifecycle,
    and Exponential Moving Average (EMA) velocity smoothing.
    """
    _id_counter = 1
    dt: float = 1.0 / 25.0
    ema_alpha: float = 0.35

    @classmethod
    def reset_id_counter(cls, start: int = 1):
        cls._id_counter = start

    def __init__(
        self,
        bbox: List[float],
        confidence: float,
        foot_point: Tuple[float, float],
        dt: float = 1.0 / 25.0,
        ema_alpha: float = 0.35
    ):
        self.track_id = SingleTrack._id_counter
        SingleTrack._id_counter += 1

        # Defensive sanitize bbox
        clean_bbox = [0.0, 0.0, 10.0, 10.0]
        if bbox is not None and len(bbox) >= 4:
            clean_bbox = [
                float(b) if np.isfinite(b) else 0.0 for b in bbox[:4]
            ]
        self.bbox = clean_bbox

        conf = float(confidence) if np.isfinite(confidence) else 1.0
        self.confidence = max(0.0, min(1.0, conf))

        # Defensive sanitize foot_point
        fx = float(foot_point[0]) if (foot_point is not None and len(foot_point) >= 2 and np.isfinite(foot_point[0])) else ((self.bbox[0] + self.bbox[2]) / 2.0)
        fy = float(foot_point[1]) if (foot_point is not None and len(foot_point) >= 2 and np.isfinite(foot_point[1])) else self.bbox[3]
        self.foot_point = (fx, fy)

        self.dt = float(dt) if (dt is not None and np.isfinite(dt) and dt > 1e-4) else (1.0 / 25.0)
        self.ema_alpha = float(ema_alpha) if (ema_alpha is not None and np.isfinite(ema_alpha)) else 0.35
        self.world_pos: Optional[Tuple[float, float]] = None

        # State lifecycle
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # Motion trail history
        self.trail: List[Tuple[int, int]] = [(int(self.foot_point[0]), int(self.foot_point[1]))]
        self.world_trail: List[Tuple[float, float]] = []

        # Real-world metric velocity & speed (EMA smoothed, alpha = 0.35)
        self.metric_vx: float = 0.0
        self.metric_vy: float = 0.0
        self.metric_speed: float = 0.0
        self._raw_vx: float = 0.0
        self._raw_vy: float = 0.0

        # Pixel-space velocity (EMA smoothed)
        self.pixel_vx: float = 0.0
        self.pixel_vy: float = 0.0

        # Pose & Kinetic Attributes
        self.keypoints: Optional[np.ndarray] = None
        self.posture: str = "standing"
        self.posture_history: List[str] = [self.posture]
        self.threat_level: int = 1
        self.pose_indicators: List[str] = []

        # Ground/Image Kalman Filter
        self.kf = KalmanFilter2D(dt=self.dt, process_noise_std=0.8, measurement_noise_std=0.3)
        self.kf.initialize(self.foot_point[0], self.foot_point[1])

    def predict(self):
        """Advances Kalman filter estimate."""
        old_x, old_y = self.foot_point
        pred_x, pred_y = self.kf.predict()
        self.foot_point = (pred_x, pred_y)

        # Shift bounding box by Kalman displacement to maintain IoU overlap
        dx = pred_x - old_x
        dy = pred_y - old_y
        if np.isfinite(dx) and np.isfinite(dy):
            self.bbox = [self.bbox[0] + dx, self.bbox[1] + dy, self.bbox[2] + dx, self.bbox[3] + dy]

        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        bbox: List[float],
        confidence: float,
        foot_point: Tuple[float, float],
        world_pt: Optional[Tuple[float, float]] = None,
        keypoints: Optional[np.ndarray] = None,
        posture_info: Optional[Dict[str, Any]] = None
    ):
        """Incorporates new matched detection with EMA velocity smoothing."""
        if bbox is not None and len(bbox) >= 4:
            self.bbox = [float(b) if np.isfinite(b) else self.bbox[i] for i, b in enumerate(bbox[:4])]
        
        conf = float(confidence) if np.isfinite(confidence) else self.confidence
        self.confidence = max(0.0, min(1.0, conf))

        fx = float(foot_point[0]) if (foot_point is not None and len(foot_point) >= 2 and np.isfinite(foot_point[0])) else self.foot_point[0]
        fy = float(foot_point[1]) if (foot_point is not None and len(foot_point) >= 2 and np.isfinite(foot_point[1])) else self.foot_point[1]
        
        self.kf.update(fx, fy)
        self.foot_point = self.kf.position

        self.hits += 1
        self.time_since_update = 0

        # Update pixel trail
        px_pt = (int(self.foot_point[0]), int(self.foot_point[1]))
        self.trail.append(px_pt)
        if len(self.trail) > 40:
            self.trail.pop(0)

        # Smooth pixel velocity from Kalman filter using EMA
        raw_kf_vx, raw_kf_vy = self.kf.velocity
        if not np.isfinite(raw_kf_vx):
            raw_kf_vx = 0.0
        if not np.isfinite(raw_kf_vy):
            raw_kf_vy = 0.0

        if self.hits <= 2:
            self.pixel_vx = raw_kf_vx
            self.pixel_vy = raw_kf_vy
        else:
            self.pixel_vx = self.ema_alpha * raw_kf_vx + (1.0 - self.ema_alpha) * self.pixel_vx
            self.pixel_vy = self.ema_alpha * raw_kf_vy + (1.0 - self.ema_alpha) * self.pixel_vy

        # Pixel-space velocity deadband (< 1.2 px/frame eliminates stationary bounding-box jitter)
        if float(np.hypot(self.pixel_vx, self.pixel_vy)) < 1.2:
            self.pixel_vx = 0.0
            self.pixel_vy = 0.0

        # Update world coordinates and real-world metric velocity with EMA smoothing
        if world_pt is not None and len(world_pt) >= 2 and np.isfinite(world_pt[0]) and np.isfinite(world_pt[1]):
            clean_world_pt = (float(world_pt[0]), float(world_pt[1]))
            self.world_pos = clean_world_pt
            self.world_trail.append(clean_world_pt)
            if len(self.world_trail) > 40:
                self.world_trail.pop(0)

            if len(self.world_trail) >= 2:
                p1 = self.world_trail[-2]
                p2 = self.world_trail[-1]
                step_dt = getattr(self, "dt", 1.0 / 25.0)
                if step_dt is None or not np.isfinite(step_dt) or step_dt <= 1e-4:
                    step_dt = 1.0 / 25.0

                raw_vx = (p2[0] - p1[0]) / step_dt
                raw_vy = (p2[1] - p1[1]) / step_dt
                if not (np.isfinite(raw_vx) and np.isfinite(raw_vy)):
                    raw_vx, raw_vy = 0.0, 0.0

                # Zero-Velocity Deadband Filtering (ZUPT):
                # Clamp sensor noise, bounding-box quantization, and micro-movements below 0.8 km/h (~0.22 m/s)
                SPEED_DEADBAND_KMH = 0.8
                raw_spd_kmh = float(np.hypot(raw_vx, raw_vy) * 3.6)
                if raw_spd_kmh < SPEED_DEADBAND_KMH:
                    raw_vx = 0.0
                    raw_vy = 0.0

                self._raw_vx = raw_vx
                self._raw_vy = raw_vy

                # Exponential Moving Average (EMA) smoothing: alpha = 0.35
                if len(self.world_trail) == 2:
                    self.metric_vx = raw_vx
                    self.metric_vy = raw_vy
                else:
                    self.metric_vx = self.ema_alpha * raw_vx + (1.0 - self.ema_alpha) * self.metric_vx
                    self.metric_vy = self.ema_alpha * raw_vy + (1.0 - self.ema_alpha) * self.metric_vy

                spd = float(np.hypot(self.metric_vx, self.metric_vy) * 3.6)
                if spd < SPEED_DEADBAND_KMH:
                    self.metric_speed = 0.0
                    self.metric_vx = 0.0
                    self.metric_vy = 0.0
                else:
                    self.metric_speed = min(35.0, spd if np.isfinite(spd) else 0.0)

        # Update pose attributes with temporal majority smoothing
        if keypoints is not None:
            self.keypoints = np.asarray(keypoints, dtype=np.float32)
        if posture_info is not None:
            raw_p = posture_info.get("posture", self.posture)
            self.posture_history.append(raw_p)
            if len(self.posture_history) > 6:
                self.posture_history.pop(0)

            # High threat postures (climbing, crawling, fallen) take priority if observed in >= 2 frames
            threats = [p for p in self.posture_history if p in ("climbing", "crawling", "fallen")]
            if len(threats) >= 2:
                from collections import Counter
                self.posture = Counter(threats).most_common(1)[0][0]
            else:
                from collections import Counter
                self.posture = Counter(self.posture_history).most_common(1)[0][0]

            self.threat_level = posture_info.get("threat_level", self.threat_level)
            self.pose_indicators = posture_info.get("indicators", self.pose_indicators)

    @property
    def velocity(self) -> Tuple[float, float]:
        if self.hits <= 1:
            return self.kf.velocity
        return (self.pixel_vx, self.pixel_vy)

    @property
    def speed(self) -> float:
        if self.hits <= 1:
            return self.kf.speed
        return float(np.hypot(self.pixel_vx, self.pixel_vy))


class MultiTargetTracker:
    """
    ByteTrack Multi-Target Association Engine.
    Combines Bounding-Box IoU with Kalman Filter state estimation.
    Solves optimal bipartite matching using two-stage Hungarian association:
    - Stage 1: High-confidence detections (conf >= 0.50) matched to confirmed tracks.
    - Stage 2: Low-confidence detections (0.15 <= conf < 0.50) matched to remaining unconfirmed/lost tracks
      to maintain identity through heavy occlusion or motion blur.
    """
    def __init__(
        self,
        min_hits: int = 2,
        max_age: int = 15,
        iou_threshold: float = 0.3,
        dt: float = 1.0 / 25.0,
        high_conf_threshold: float = 0.50,
        low_conf_threshold: float = 0.15,
        ema_alpha: float = 0.35
    ):
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.dt = dt
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.ema_alpha = ema_alpha
        self.tracks: List[SingleTrack] = []

    def _update_single_track(
        self,
        trk: SingleTrack,
        det: Dict[str, Any],
        homography_transformer=None,
        pose_classifier=None
    ):
        """Helper to transform coordinates and incorporate detection into a track with ankle ground refinement."""
        foot_pt = det.get("foot_point", (0.0, 0.0))
        keypoints = det.get("keypoints")

        # Ankle-Refined Ground Contact Point:
        # True physical ground contact is the midpoint of the ankles (COCO joints 15 & 16),
        # which eliminates bounding box bottom noise from ground shadows or clothing.
        if keypoints is not None:
            try:
                kpts_arr = np.asarray(keypoints, dtype=np.float32)
                if kpts_arr.shape[0] >= 17:
                    l_ank = kpts_arr[15]
                    r_ank = kpts_arr[16]
                    l_conf = l_ank[2] if len(l_ank) >= 3 else 1.0
                    r_conf = r_ank[2] if len(r_ank) >= 3 else 1.0
                    if l_conf >= 0.35 and r_conf >= 0.35:
                        foot_pt = (float((l_ank[0] + r_ank[0]) / 2.0), float((l_ank[1] + r_ank[1]) / 2.0))
                    elif l_conf >= 0.35:
                        foot_pt = (float(l_ank[0]), float(l_ank[1]))
                    elif r_conf >= 0.35:
                        foot_pt = (float(r_ank[0]), float(r_ank[1]))
            except Exception:
                pass

        world_pt = None
        if homography_transformer and homography_transformer.is_calibrated:
            try:
                world_pt = homography_transformer.image_to_world(
                    foot_pt[0], foot_pt[1]
                )
            except Exception:
                world_pt = None

        posture_info = det.get("posture_info")
        if posture_info is None and keypoints is not None:
            # Classify posture via pose_classifier or standalone helper
            speed_mps = (trk.metric_speed / 3.6) if trk.world_pos is not None else 0.0
            if pose_classifier is not None:
                try:
                    posture_info = pose_classifier.classify_pose(keypoints, speed_mps=speed_mps)
                except Exception:
                    posture_info = None
            else:
                try:
                    from cv_core.kinetics.pose_classifier import classify_keypoints_posture
                    posture_info = classify_keypoints_posture(keypoints, speed_mps=speed_mps)
                except Exception:
                    posture_info = None

        trk.update(
            bbox=det["bbox"],
            confidence=det["confidence"],
            foot_point=foot_pt,
            world_pt=world_pt,
            keypoints=keypoints,
            posture_info=posture_info
        )

    def update(
        self,
        detections: Optional[List[Dict[str, Any]]] = None,
        homography_transformer=None,
        pose_classifier=None
    ) -> List[Dict[str, Any]]:
        """
        ByteTrack Two-Stage Association Update:
            detections: List of dicts with:
                'bbox': [x1, y1, x2, y2]
                'confidence': float
                'foot_point': (foot_x, foot_y)
                'keypoints': Optional (17, 2) or (17, 3) keypoint array
        """
        if detections is None:
            detections = []

        # Validate and sanitize input detections to prevent NaN/corrupt input crashes
        cleaned_detections: List[Dict[str, Any]] = []
        for det in detections:
            if not isinstance(det, dict) or "bbox" not in det:
                continue
            raw_box = det["bbox"]
            if raw_box is None or len(raw_box) < 4:
                continue
            try:
                box = [float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3])]
            except (ValueError, TypeError):
                continue
            if not all(np.isfinite(c) for c in box):
                continue

            conf = float(det.get("confidence", 1.0))
            if not np.isfinite(conf):
                conf = 0.0

            fp = det.get("foot_point")
            if fp is None or len(fp) < 2 or not (np.isfinite(fp[0]) and np.isfinite(fp[1])):
                fp = ((box[0] + box[2]) / 2.0, box[3])
            else:
                fp = (float(fp[0]), float(fp[1]))

            clean_d = {
                "bbox": box,
                "confidence": conf,
                "foot_point": fp
            }
            if "keypoints" in det:
                clean_d["keypoints"] = det["keypoints"]
            if "posture_info" in det:
                clean_d["posture_info"] = det["posture_info"]
            cleaned_detections.append(clean_d)

        detections = cleaned_detections

        # 1. Predict new locations of all existing tracks
        for trk in self.tracks:
            trk.predict()

        # 2. Partition detections into High-Confidence and Low-Confidence bands
        high_dets: List[Dict[str, Any]] = []
        low_dets: List[Dict[str, Any]] = []
        for det in detections:
            conf = float(det.get("confidence", 1.0))
            if conf >= self.high_conf_threshold:
                high_dets.append(det)
            elif conf >= self.low_conf_threshold:
                low_dets.append(det)

        # 3. Partition tracks into confirmed, unconfirmed (tentative), and lost
        confirmed_tracks = [t for t in self.tracks if t.state == TrackState.CONFIRMED]
        unconfirmed_tracks = [t for t in self.tracks if t.state == TrackState.TENTATIVE]
        lost_tracks = [t for t in self.tracks if t.state == TrackState.LOST]

        # =========================================================================
        # STAGE 1: Match confirmed tracks with high-confidence detections
        # =========================================================================
        matches_s1, unmatched_confirmed_idx, unmatched_high_idx = linear_assignment_iou(
            confirmed_tracks, high_dets, self.iou_threshold
        )
        for r, c in matches_s1:
            self._update_single_track(
                confirmed_tracks[r], high_dets[c],
                homography_transformer=homography_transformer,
                pose_classifier=pose_classifier
            )

        # Match remaining high-confidence detections with unconfirmed (tentative) tracks
        rem_high_dets = [high_dets[i] for i in unmatched_high_idx]
        matches_unconf, unmatched_unconf_idx, unmatched_rem_high_idx = linear_assignment_iou(
            unconfirmed_tracks, rem_high_dets, self.iou_threshold
        )
        for r, c in matches_unconf:
            self._update_single_track(
                unconfirmed_tracks[r], rem_high_dets[c],
                homography_transformer=homography_transformer,
                pose_classifier=pose_classifier
            )

        # Match still remaining high-confidence detections with lost tracks (recovering lost targets)
        lost_candidates_high = [rem_high_dets[i] for i in unmatched_rem_high_idx]
        matches_lost_high, unmatched_lost_idx, unmatched_final_high_idx = linear_assignment_iou(
            lost_tracks, lost_candidates_high, self.iou_threshold
        )
        for r, c in matches_lost_high:
            self._update_single_track(
                lost_tracks[r], lost_candidates_high[c],
                homography_transformer=homography_transformer,
                pose_classifier=pose_classifier
            )

        # =========================================================================
        # STAGE 2: Match remaining unconfirmed/lost tracks with low-confidence detections
        # =========================================================================
        # Remaining tracks = unmatched confirmed from stage 1 + unmatched tentative + unmatched lost
        unmatched_confirmed_tracks = [confirmed_tracks[i] for i in unmatched_confirmed_idx]
        unmatched_unconf_tracks = [unconfirmed_tracks[i] for i in unmatched_unconf_idx]
        unmatched_lost_tracks = [lost_tracks[i] for i in unmatched_lost_idx]

        remaining_tracks_stage2 = (
            unmatched_confirmed_tracks + unmatched_unconf_tracks + unmatched_lost_tracks
        )

        matches_s2, unmatched_s2_tracks_idx, _ = linear_assignment_iou(
            remaining_tracks_stage2, low_dets, self.iou_threshold
        )
        for r, c in matches_s2:
            self._update_single_track(
                remaining_tracks_stage2[r], low_dets[c],
                homography_transformer=homography_transformer,
                pose_classifier=pose_classifier
            )
        # Note: Unmatched low-confidence detections are discarded to prevent false positive tracks.

        # =========================================================================
        # Track Initialization: Only remaining high-confidence detections spawn new tracks
        # =========================================================================
        for i in unmatched_final_high_idx:
            det = lost_candidates_high[i]
            foot_pt = det.get("foot_point", (0.0, 0.0))
            if det.get("keypoints") is not None:
                try:
                    kpts_arr = np.asarray(det["keypoints"], dtype=np.float32)
                    if kpts_arr.shape[0] >= 17:
                        l_ank, r_ank = kpts_arr[15], kpts_arr[16]
                        l_conf = l_ank[2] if len(l_ank) >= 3 else 1.0
                        r_conf = r_ank[2] if len(r_ank) >= 3 else 1.0
                        if l_conf >= 0.35 and r_conf >= 0.35:
                            foot_pt = (float((l_ank[0] + r_ank[0]) / 2.0), float((l_ank[1] + r_ank[1]) / 2.0))
                        elif l_conf >= 0.35:
                            foot_pt = (float(l_ank[0]), float(l_ank[1]))
                        elif r_conf >= 0.35:
                            foot_pt = (float(r_ank[0]), float(r_ank[1]))
                except Exception:
                    pass

            new_track = SingleTrack(
                bbox=det["bbox"],
                confidence=det["confidence"],
                foot_point=foot_pt,
                dt=self.dt,
                ema_alpha=self.ema_alpha
            )
            if det.get("keypoints") is not None:
                new_track.keypoints = np.asarray(det["keypoints"], dtype=np.float32)
            if homography_transformer and homography_transformer.is_calibrated:
                try:
                    new_track.world_pos = homography_transformer.image_to_world(
                        foot_pt[0], foot_pt[1]
                    )
                except Exception:
                    pass
            self.tracks.append(new_track)

        # =========================================================================
        # Lifecycle Maintenance & Output Construction
        # =========================================================================
        active_tracks: List[SingleTrack] = []
        output_tracks: List[Dict[str, Any]] = []

        for trk in self.tracks:
            # Promote TENTATIVE or recover LOST tracks upon successful matching
            if trk.time_since_update == 0:
                if trk.state == TrackState.TENTATIVE and trk.hits >= self.min_hits:
                    trk.state = TrackState.CONFIRMED
                elif trk.state == TrackState.LOST:
                    trk.state = TrackState.CONFIRMED

            # Check track loss/deletion
            if trk.time_since_update > self.max_age:
                trk.state = TrackState.DELETED
            elif trk.time_since_update > 0:
                trk.state = TrackState.LOST

            if trk.state != TrackState.DELETED:
                active_tracks.append(trk)

            # Output only confirmed tracks (or active tentative if fresh)
            if trk.state == TrackState.CONFIRMED or (trk.state == TrackState.TENTATIVE and trk.hits >= 2):
                vx, vy = (trk.metric_vx, trk.metric_vy) if trk.world_pos is not None else trk.velocity
                if not (np.isfinite(vx) and np.isfinite(vy)):
                    vx, vy = 0.0, 0.0
                speed_kmh = trk.metric_speed if trk.world_pos is not None else float(trk.speed * 3.6)
                if not np.isfinite(speed_kmh) or speed_kmh < 0.8:
                    speed_kmh = 0.0
                track_info: Dict[str, Any] = {
                    "track_id": trk.track_id,
                    "bbox": trk.bbox,
                    "confidence": trk.confidence,
                    "foot_point": trk.foot_point,
                    "trail": list(trk.trail),
                    "world_pos": trk.world_pos,
                    "world_trail": list(trk.world_trail),
                    "vx": vx,
                    "vy": vy,
                    "speed": speed_kmh,
                    "state": trk.state.value,
                    "posture": trk.posture,
                    "threat_level": trk.threat_level,
                    "indicators": list(trk.pose_indicators)
                }
                if trk.keypoints is not None:
                    track_info["keypoints"] = trk.keypoints
                output_tracks.append(track_info)

        self.tracks = active_tracks
        return output_tracks
