import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
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
        self.world_bounds = tuple(world_bounds) if world_bounds is not None else (0.0, 0.0, 20.0, 20.0)
        if (len(self.world_bounds) != 4 or not np.all(np.isfinite(self.world_bounds))
                or self.world_bounds[2] <= self.world_bounds[0] or self.world_bounds[3] <= self.world_bounds[1]):
            raise ValueError('World bounds must be finite with positive width and height')
        self.unit = unit

        if H_matrix is not None:
            self.set_homography(H_matrix)

    def set_homography(self, H: np.ndarray):
        """Validates and sets the 3x3 homography matrix and computes its inverse."""
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got shape {H.shape}")

        # Fix #10: Reject non-finite matrices before any other check
        if not np.all(np.isfinite(H)):
            raise ValueError("Homography matrix contains non-finite values (NaN or Inf).")

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
        if len(image_points) != len(world_points):
            raise ValueError('Calibration point lists must have equal lengths')
        if not np.all(np.isfinite(image_points)) or not np.all(np.isfinite(world_points)):
            raise ValueError('Calibration coordinates must be finite')
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

        if not (np.isfinite(u) and np.isfinite(v)):
            raise ValueError(f"Non-finite image coordinates: ({u}, {v})")

        vec = np.array([float(u), float(v), 1.0], dtype=np.float64)
        projected = self.H @ vec
        
        if np.isclose(projected[2], 0.0, atol=1e-9):
            raise ZeroDivisionError("Projective singularity encountered (point at infinity).")

        X = float(projected[0] / projected[2])
        Y = float(projected[1] / projected[2])
        return (X, Y)

    def is_image_point_in_front_of_camera(self, u: float, v: float) -> bool:
        """Checks whether the image coordinate projects to the valid half-plane in front of camera."""
        if not self.is_calibrated or not (np.isfinite(u) and np.isfinite(v)):
            return False
        vec = np.array([float(u), float(v), 1.0], dtype=np.float64)
        projected = self.H @ vec
        return float(projected[2]) > 1e-9

    def world_to_image(self, X: float, Y: float, check_camera_plane: bool = False) -> Tuple[float, float]:
        """
        Reprojects ground coordinates (X, Y) back into camera pixel coordinates (u, v).
        [u, v, 1]^T ~ H_inv * [X, Y, 1]^T
        
        If check_camera_plane is True, raises ValueError if the point lies behind the camera plane.
        """
        if not self.is_calibrated:
            raise RuntimeError("HomographyTransformer has not been calibrated.")

        if not (np.isfinite(X) and np.isfinite(Y)):
            raise ValueError(f"Non-finite world coordinates: ({X}, {Y})")

        vec = np.array([float(X), float(Y), 1.0], dtype=np.float64)
        reprojected = self.H_inv @ vec

        if np.isclose(reprojected[2], 0.0, atol=1e-9):
            raise ZeroDivisionError("Projective singularity encountered in inverse projection.")

        if check_camera_plane and reprojected[2] < 0.0:
            raise ValueError(f"World point ({X}, {Y}) is behind camera plane (w={reprojected[2]:.4f}).")

        u = float(reprojected[0] / reprojected[2])
        v = float(reprojected[1] / reprojected[2])
        return (u, v)

    def is_world_point_in_front_of_camera(self, X: float, Y: float) -> bool:
        """Checks whether the world coordinate lies strictly in front of the camera projection plane."""
        if not self.is_calibrated or not (np.isfinite(X) and np.isfinite(Y)):
            return False
        vec = np.array([float(X), float(Y), 1.0], dtype=np.float64)
        reprojected = self.H_inv @ vec
        return float(reprojected[2]) > 1e-9

    def compute_reprojection_error(
        self,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]]
    ) -> float:
        """Calculates root mean squared reprojection error in world units (meters)."""
        if len(image_points) != len(world_points):
            raise ValueError('Calibration point lists must have equal lengths')
        if len(image_points) == 0:
            return float("inf")
        errors = []
        for (u, v), (X_gt, Y_gt) in zip(image_points, world_points):
            try:
                X_pred, Y_pred = self.image_to_world(u, v)
                dist = np.hypot(X_pred - X_gt, Y_pred - Y_gt)
                if np.isfinite(dist):
                    errors.append(dist)
            except Exception:
                continue
        # Fix #10: return infinity when no valid projections succeed rather than silently returning 0.0
        if not errors:
            return float("inf")
        return float(np.sqrt(np.mean(np.array(errors) ** 2)))

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
        zero_vel = {"speed_mps": 0.0, "speed_kmh": 0.0, "direction_deg": 0.0, "vx": 0.0, "vy": 0.0}
        if not history_world_pts or len(history_world_pts) < 2:
            return zero_vel

        # Use recent temporal window (last ~0.5 to 1.5s) to smooth jitter
        p1 = history_world_pts[0]
        p2 = history_world_pts[-1]

        if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2:
            return zero_vel

        if not (np.isfinite(p1[0]) and np.isfinite(p1[1]) and np.isfinite(p2[0]) and np.isfinite(p2[1])):
            return zero_vel

        if len(p1) >= 3 and len(p2) >= 3:
            t1, t2 = p1[2], p2[2]
            if np.isfinite(t1) and np.isfinite(t2):
                dt = t2 - t1
            else:
                dt = 0.0
        else:
            fps_val = float(fps) if (fps is not None and np.isfinite(fps) and fps > 1e-3) else 30.0
            dt = (len(history_world_pts) - 1) * (1.0 / fps_val)

        if not np.isfinite(dt) or dt <= 1e-4:
            return zero_vel

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if np.isclose(dx, 0.0, atol=1e-6) and np.isclose(dy, 0.0, atol=1e-6):
            return zero_vel

        vx = dx / dt
        vy = dy / dt
        speed_mps = float(np.hypot(vx, vy))
        if not np.isfinite(speed_mps):
            return zero_vel

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

    @staticmethod
    def _clip_line_to_viewport(
        x1: float, y1: float, x2: float, y2: float,
        u_min: float, v_min: float, u_max: float, v_max: float
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Liang-Barsky parametric 2D line clipping algorithm.
        Clips line segment from (x1, y1) to (x2, y2) to viewport [u_min, v_min, u_max, v_max].
        Returns integer pixel coordinate endpoints or None if segment is entirely outside.
        """
        if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
            return None
        dx = x2 - x1
        dy = y2 - y1
        p = [-dx, dx, -dy, dy]
        q = [x1 - u_min, u_max - x1, y1 - v_min, v_max - y1]
        u1, u2 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if pi == 0:
                if qi < 0:
                    return None
            else:
                t = qi / pi
                if pi < 0:
                    if t > u2:
                        return None
                    if t > u1:
                        u1 = t
                elif pi > 0:
                    if t < u1:
                        return None
                    if t < u2:
                        u2 = t
        if u1 > u2:
            return None
        cx1 = x1 + u1 * dx
        cy1 = y1 + u1 * dy
        cx2 = x1 + u2 * dx
        cy2 = y1 + u2 * dy
        return ((int(round(cx1)), int(round(cy1))), (int(round(cx2)), int(round(cy2))))

    def project_ground_grid_to_image(
        self,
        image_shape: Tuple[int, int],
        grid_step_m: float = 2.0,
        max_visible_y: Optional[float] = None
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Projects real-world ground grid lines (X and Y in meters) into camera image pixel coordinates.
        Clips and clamps lines to the valid visible floor plane in front of the camera so grid lines
        do not shoot up into walls, doors, or ceilings.

        Args:
            image_shape: (height, width) of the target camera frame.
            grid_step_m: Step size in meters between ground grid lines.
            max_visible_y: Optional upper bound in meters for visible ground depth (Y).
                           If None, clamps to the visible floor horizon limit (defaulting
                           to 6.0m when default 20m bounds are present).

        Returns:
            List of ((u1, v1), (u2, v2)) pixel coordinate segments to render on perspective view.
        """
        if not self.is_calibrated:
            return []

        h, w = image_shape[:2]
        x_min, y_min, x_max, y_max = self.world_bounds
        step = float(grid_step_m) if (grid_step_m is not None and grid_step_m > 1e-3) else 2.0

        if max_visible_y is not None:
            effective_y_max = min(y_max, max(y_min, float(max_visible_y)))
        else:
            effective_y_max = min(y_max, 6.0) if y_max >= 20.0 else y_max

        segments = []

        # X-lines (constant X, varying Y across visible floor plane)
        x_vals = np.arange(x_min, x_max + 1e-5, step)
        for x in x_vals:
            try:
                u1, v1 = self.world_to_image(x, y_min)
                u2, v2 = self.world_to_image(x, effective_y_max)
                clipped = self._clip_line_to_viewport(u1, v1, u2, v2, 0.0, 0.0, float(w), float(h))
                if clipped is not None:
                    segments.append(clipped)
            except Exception:
                continue

        # Y-lines (constant Y, varying X across visible floor width)
        y_vals = np.arange(y_min, effective_y_max + 1e-5, step)
        for y in y_vals:
            try:
                u1, v1 = self.world_to_image(x_min, y)
                u2, v2 = self.world_to_image(x_max, y)
                clipped = self._clip_line_to_viewport(u1, v1, u2, v2, 0.0, 0.0, float(w), float(h))
                if clipped is not None:
                    segments.append(clipped)
            except Exception:
                continue

        return segments

    def save_calibration(self, filepath: Union[Path, str], source=None, image_size=None):
        """Saves calibration matrix and parameters to a JSON file."""
        if not self.is_calibrated:
            raise RuntimeError("Cannot save uncalibrated transformer.")
        
        data = {
            "homography_matrix": self.H.tolist(),
            "world_bounds": list(self.world_bounds),
            "unit": self.unit
        }
        if source is not None:
            data['source'] = str(source)
        if image_size is not None:
            data['image_size'] = list(image_size)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Calibration saved to {filepath}")

    @classmethod
    def load_calibration(cls, filepath: Union[Path, str]) -> "HomographyTransformer":
        """Loads calibration from a saved JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        H = np.array(data["homography_matrix"], dtype=np.float64)
        bounds = tuple(data.get("world_bounds", [0.0, 0.0, 20.0, 20.0]))
        unit = data.get("unit", "meters")
        return cls(H_matrix=H, world_bounds=bounds, unit=unit)

    @classmethod
    def for_source(cls, source, image_size, calibration_file=None):
        """Auto-load only camera-bound calibration; an explicit legacy file is operator-selected."""
        candidates = [Path(calibration_file)] if calibration_file else [
            Path(str(source) + '.calibration.json'), Path('data/calibration.json')]
        for path in candidates:
            if not path.exists():
                if calibration_file:
                    raise FileNotFoundError(path)
                continue
            with path.open() as f:
                data = json.load(f)
            bound_source, bound_size = data.get('source'), data.get('image_size')
            if (bound_source is None) != (bound_size is None):
                raise ValueError('Calibration source and image size must be supplied together')
            if bound_source is None:
                if not calibration_file:
                    continue
            else:
                if str(bound_source) != str(source):
                    if calibration_file:
                        raise ValueError('Calibration source does not match the selected feed')
                    continue
                if tuple(bound_size) != tuple(image_size):
                    if calibration_file:
                        raise ValueError('Calibration image size does not match the selected feed')
                    continue
            return cls.load_calibration(path)
        return cls()
