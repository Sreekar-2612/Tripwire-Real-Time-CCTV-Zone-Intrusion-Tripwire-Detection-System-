from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from cv_core.tracking.kalman_filter import KalmanFilter2D

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

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    inter_wh = np.maximum(0.0, bottom_right - top_left)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    union_area = np.maximum(union_area, 1e-6)

    return inter_area / union_area

class SingleTrack:
    """Represents a single tracked target with state estimation and lifecycle."""
    _id_counter = 1
    dt: float = 1.0 / 25.0

    def __init__(self, bbox: List[float], confidence: float, foot_point: Tuple[float, float], dt: float = 1.0/25.0):
        self.track_id = SingleTrack._id_counter
        SingleTrack._id_counter += 1

        self.bbox = list(bbox)
        self.confidence = confidence
        self.foot_point = foot_point
        self.dt = dt
        self.world_pos: Optional[Tuple[float, float]] = None

        # State lifecycle
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # Motion trail history
        self.trail: List[Tuple[int, int]] = [(int(foot_point[0]), int(foot_point[1]))]
        self.world_trail: List[Tuple[float, float]] = []
        self.metric_vx: float = 0.0
        self.metric_vy: float = 0.0
        self.metric_speed: float = 0.0

        # Ground/Image Kalman Filter
        self.kf = KalmanFilter2D(dt=dt, process_noise_std=0.8, measurement_noise_std=0.3)
        self.kf.initialize(foot_point[0], foot_point[1])

    def predict(self):
        """Advances Kalman filter estimate."""
        old_x, old_y = self.foot_point
        pred_x, pred_y = self.kf.predict()
        self.foot_point = (pred_x, pred_y)

        # Shift bounding box by Kalman displacement to maintain IoU overlap
        dx = pred_x - old_x
        dy = pred_y - old_y
        self.bbox = [self.bbox[0] + dx, self.bbox[1] + dy, self.bbox[2] + dx, self.bbox[3] + dy]

        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: List[float], confidence: float, foot_point: Tuple[float, float], world_pt: Optional[Tuple[float, float]] = None):
        """Incorporates new matched detection."""
        self.bbox = list(bbox)
        self.confidence = confidence
        self.kf.update(foot_point[0], foot_point[1])
        self.foot_point = self.kf.position

        self.hits += 1
        self.time_since_update = 0

        # Update trail
        px_pt = (int(self.foot_point[0]), int(self.foot_point[1]))
        self.trail.append(px_pt)
        if len(self.trail) > 40:
            self.trail.pop(0)

        if world_pt is not None:
            self.world_pos = world_pt
            self.world_trail.append(world_pt)
            if len(self.world_trail) > 40:
                self.world_trail.pop(0)

            if len(self.world_trail) >= 2:
                p1 = self.world_trail[-2]
                p2 = self.world_trail[-1]
                step_dt = max(0.001, getattr(self, "dt", 1.0 / 25.0))
                self.metric_vx = (p2[0] - p1[0]) / step_dt
                self.metric_vy = (p2[1] - p1[1]) / step_dt
                self.metric_speed = min(35.0, float(np.hypot(self.metric_vx, self.metric_vy) * 3.6))

    @property
    def velocity(self) -> Tuple[float, float]:
        return self.kf.velocity

    @property
    def speed(self) -> float:
        return self.kf.speed

class MultiTargetTracker:
    """
    Multi-Target Association Engine.
    Combines Bounding-Box IoU with Kalman Filter state estimation.
    Solves optimal bipartite matching using the Hungarian algorithm.
    """
    def __init__(
        self,
        min_hits: int = 2,
        max_age: int = 15,
        iou_threshold: float = 0.3,
        dt: float = 1.0 / 25.0
    ):
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.dt = dt
        self.tracks: List[SingleTrack] = []

    def update(
        self,
        detections: List[Dict[str, Any]],
        homography_transformer=None
    ) -> List[Dict[str, Any]]:
        """
        detections: List of dicts with:
            'bbox': [x1, y1, x2, y2]
            'confidence': float
            'foot_point': (foot_x, foot_y)
        """
        # 1. Predict new locations of existing tracks
        for trk in self.tracks:
            trk.predict()

        num_tracks = len(self.tracks)
        num_dets = len(detections)

        matched_tracks = set()
        matched_dets = set()

        if num_tracks > 0 and num_dets > 0:
            track_boxes = np.array([t.bbox for t in self.tracks], dtype=np.float32)
            det_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)

            # Compute IoU matrix
            iou_matrix = compute_iou_matrix(track_boxes, det_boxes)
            # Cost matrix = 1.0 - IoU
            cost_matrix = 1.0 - iou_matrix

            # Hungarian bipartite matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_tracks.add(r)
                    matched_dets.add(c)
                    det = detections[c]
                    world_pt = None
                    if homography_transformer and homography_transformer.is_calibrated:
                        try:
                            world_pt = homography_transformer.image_to_world(det["foot_point"][0], det["foot_point"][1])
                        except Exception:
                            world_pt = None

                    self.tracks[r].update(
                        bbox=det["bbox"],
                        confidence=det["confidence"],
                        foot_point=det["foot_point"],
                        world_pt=world_pt
                    )

        # 2. Create new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx not in matched_dets:
                new_track = SingleTrack(
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    foot_point=det["foot_point"],
                    dt=self.dt
                )
                if homography_transformer and homography_transformer.is_calibrated:
                    try:
                        new_track.world_pos = homography_transformer.image_to_world(det["foot_point"][0], det["foot_point"][1])
                    except Exception:
                        pass
                self.tracks.append(new_track)

        # 3. Update track lifecycle states
        active_tracks = []
        output_tracks = []

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
                speed_kmh = trk.metric_speed if trk.world_pos is not None else float(trk.speed * 3.6)
                output_tracks.append({
                    "track_id": trk.track_id,
                    "bbox": trk.bbox,
                    "confidence": trk.confidence,
                    "foot_point": trk.foot_point,
                    "trail": list(trk.trail),
                    "world_pos": trk.world_pos,
                    "world_trail": list(trk.world_trail),
                    "vx": vx,
                    "vy": vy,
                    "speed": speed_kmh
                })

        self.tracks = active_tracks
        return output_tracks
