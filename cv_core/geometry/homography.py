import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger("tripwire.cv.homography")

class HomographyTransformer:
    """
    Planar Homography Projective Geometry Engine.
    
    Maps 2D pixel coordinates (u, v) from camera perspective onto a 2D
    real-world ground plane (X, Y) in meters, enabling:
    1. Bird's Eye View (BEV) top-down visualization.
    2. Physical metric distance & area measurements.
    3. Ground-truth velocity (m/s, km/h) estimation independent of camera tilt.
    """

    def __init__(
        self,
        H_matrix: Optional[np.ndarray] = None,
        world_bounds: Optional[Tuple[float, float, float, float]] = None,
        unit: str = "meters"
    ):
        """
        world_bounds: (x_min, y_min, x_max, y_max) defining metric ground extent in meters.
        """
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.world_bounds = world_bounds or (0.0, 0.0, 20.0, 20.0)
        self.unit = unit

        if H_matrix is not None:
            self.set_homography(H_matrix)

    def set_homography(self, H: np.ndarray):
        """Validates and sets the 3x3 homography matrix and computes its inverse."""
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got shape {H.shape}")
        
        det = np.linalg.det(H)
        if np.isclose(det, 0.0, atol=1e-9):
            raise ValueError("Singular homography matrix cannot be inverted.")

        # Normalize so H[2, 2] = 1.0 if non-zero
        if not np.isclose(H[2, 2], 0.0):
            H = H / H[2, 2]

        self.H = H
        self.H_inv = np.linalg.inv(H)

    @classmethod
    def from_point_correspondences(
        cls,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]],
        world_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> "HomographyTransformer":
        """
        Computes 3x3 homography using Hartley-normalized Direct Linear Transformation (DLT)
        or RANSAC if >= 4 correspondences are provided.
        """
        if len(image_points) < 4 or len(world_points) < 4:
            raise ValueError(f"At least 4 point correspondences required, got {len(image_points)}")

        src_pts = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.asarray(world_points, dtype=np.float32).reshape(-1, 1, 2)

        # Compute Homography via OpenCV DLT / RANSAC
        if len(image_points) == 4:
            H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
        else:
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0.5)

        if H is None:
            raise RuntimeError("Failed to compute valid homography matrix from point correspondences.")

        return cls(H_matrix=H, world_bounds=world_bounds)

    @property
    def is_calibrated(self) -> bool:
        return self.H is not None and self.H_inv is not None

    def image_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        Projects an image pixel (u, v) onto the metric ground plane (X, Y) in meters.
        [X, Y, 1]^T ~ H * [u, v, 1]^T
        """
        if not self.is_calibrated:
            raise RuntimeError("HomographyTransformer has not been calibrated.")

        vec = np.array([u, v, 1.0], dtype=np.float64)
        projected = self.H @ vec
        
        if np.isclose(projected[2], 0.0, atol=1e-9):
            raise ZeroDivisionError("Projective singularity encountered (point at infinity).")

        X = float(projected[0] / projected[2])
        Y = float(projected[1] / projected[2])
        return (X, Y)

    def world_to_image(self, X: float, Y: float) -> Tuple[float, float]:
        """
        Reprojects ground coordinates (X, Y) back into camera pixel coordinates (u, v).
        [u, v, 1]^T ~ H_inv * [X, Y, 1]^T
        """
        if not self.is_calibrated:
            raise RuntimeError("HomographyTransformer has not been calibrated.")

        vec = np.array([X, Y, 1.0], dtype=np.float64)
        reprojected = self.H_inv @ vec

        if np.isclose(reprojected[2], 0.0, atol=1e-9):
            raise ZeroDivisionError("Projective singularity encountered in inverse projection.")

        u = float(reprojected[0] / reprojected[2])
        v = float(reprojected[1] / reprojected[2])
        return (u, v)

    def compute_reprojection_error(
        self,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]]
    ) -> float:
        """Calculates root mean squared reprojection error in world units (meters)."""
        errors = []
        for (u, v), (X_gt, Y_gt) in zip(image_points, world_points):
            X_pred, Y_pred = self.image_to_world(u, v)
            dist = np.hypot(X_pred - X_gt, Y_pred - Y_gt)
            errors.append(dist)
        return float(np.mean(errors)) if errors else 0.0

    def compute_velocity(
        self,
        history_world_pts: List[Any],
        fps: float = 30.0
    ) -> Dict[str, float]:
        """
        Estimates ground plane velocity vector and speed from a temporal trajectory:
        history_world_pts: List of (X_meters, Y_meters) or (X_meters, Y_meters, timestamp_sec).
        
        Returns:
            speed_mps: Speed in meters per second
            speed_kmh: Speed in km/h
            direction_deg: Direction in degrees [0, 360) where 0 is +X (East)
            vx: X-velocity (m/s)
            vy: Y-velocity (m/s)
        """
        if len(history_world_pts) < 2:
            return {"speed_mps": 0.0, "speed_kmh": 0.0, "direction_deg": 0.0, "vx": 0.0, "vy": 0.0}

        # Use recent temporal window (last ~0.5 to 1.5s) to smooth jitter
        p1 = history_world_pts[0]
        p2 = history_world_pts[-1]

        if len(p1) >= 3 and len(p2) >= 3:
            dt = p2[2] - p1[2]
        else:
            dt = (len(history_world_pts) - 1) * (1.0 / max(1.0, fps))

        if dt <= 1e-4:
            return {"speed_mps": 0.0, "speed_kmh": 0.0, "direction_deg": 0.0, "vx": 0.0, "vy": 0.0}

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        vx = dx / dt
        vy = dy / dt
        speed_mps = float(np.hypot(vx, vy))
        speed_kmh = float(speed_mps * 3.6)

        angle_rad = np.arctan2(vy, dx)
        direction_deg = float((np.degrees(angle_rad) + 360) % 360)

        return {
            "speed_mps": round(speed_mps, 2),
            "speed_kmh": round(speed_kmh, 2),
            "direction_deg": round(direction_deg, 1),
            "vx": round(vx, 3),
            "vy": round(vy, 3)
        }

    def project_ground_grid_to_image(
        self,
        image_shape: Tuple[int, int],
        grid_step_m: float = 2.0
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Projects real-world ground grid lines (X and Y in meters) into camera image pixel coordinates.
        Returns list of ((u1, v1), (u2, v2)) line segments to render on perspective view.
        """
        if not self.is_calibrated:
            return []

        h, w = image_shape[:2]
        x_min, y_min, x_max, y_max = self.world_bounds
        segments = []

        # X-lines (constant X, varying Y)
        x_vals = np.arange(x_min, x_max + grid_step_m, grid_step_m)
        for x in x_vals:
            try:
                u1, v1 = self.world_to_image(x, y_min)
                u2, v2 = self.world_to_image(x, y_max)
                if 0 <= u1 < w * 2 and 0 <= v1 < h * 2 and 0 <= u2 < w * 2 and 0 <= v2 < h * 2:
                    segments.append(((int(u1), int(v1)), (int(u2), int(v2))))
            except Exception:
                continue

        # Y-lines (constant Y, varying X)
        y_vals = np.arange(y_min, y_max + grid_step_m, grid_step_m)
        for y in y_vals:
            try:
                u1, v1 = self.world_to_image(x_min, y)
                u2, v2 = self.world_to_image(x_max, y)
                if 0 <= u1 < w * 2 and 0 <= v1 < h * 2 and 0 <= u2 < w * 2 and 0 <= v2 < h * 2:
                    segments.append(((int(u1), int(v1)), (int(u2), int(v2))))
            except Exception:
                continue

        return segments

    def save_calibration(self, filepath: Path or str):
        """Saves calibration matrix and parameters to a JSON file."""
        if not self.is_calibrated:
            raise RuntimeError("Cannot save uncalibrated transformer.")
        
        data = {
            "homography_matrix": self.H.tolist(),
            "world_bounds": list(self.world_bounds),
            "unit": self.unit
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Calibration saved to {filepath}")

    @classmethod
    def load_calibration(cls, filepath: Path or str) -> "HomographyTransformer":
        """Loads calibration from a saved JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        H = np.array(data["homography_matrix"], dtype=np.float64)
        bounds = tuple(data.get("world_bounds", [0.0, 0.0, 20.0, 20.0]))
        unit = data.get("unit", "meters")
        return cls(H_matrix=H, world_bounds=bounds, unit=unit)
