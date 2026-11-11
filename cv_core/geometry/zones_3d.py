import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.validation import make_valid

from cv_core.geometry.homography import HomographyTransformer

logger = logging.getLogger("tripwire.cv.zones_3d")

class MetricZone3D:
    """
    Represents a security zone or tripwire defined in real-world metric space (meters).
    Points are (X, Y) on the ground plane.
    """
    def __init__(
        self,
        zone_id: str,
        name: str,
        zone_type: str,  # 'polygon' or 'tripwire'
        world_coordinates: List[Tuple[float, float]],  # List of (X, Y) in meters
        armed: bool = True,
        color: str = "#EF4444",
        min_velocity_threshold: float = 0.0  # Optional speed gating (m/s)
    ):
        self.zone_id = zone_id
        self.name = name
        self.zone_type = zone_type.lower()
        self.world_coords = world_coordinates
        self.armed = armed
        self.color = color
        self.min_velocity_threshold = min_velocity_threshold

        self.polygon: Optional[Polygon] = None
        self.line: Optional[LineString] = None

        self._build_shape()

    def _build_shape(self):
        try:
            if self.zone_type == "tripwire":
                if len(self.world_coords) >= 2:
                    self.line = LineString([(p[0], p[1]) for p in self.world_coords[:2]])
            else:
                if len(self.world_coords) >= 3:
                    poly = Polygon([(p[0], p[1]) for p in self.world_coords])
                    if not poly.is_valid:
                        poly = make_valid(poly)
                    self.polygon = poly
        except Exception as e:
            logger.error(f"Error creating metric shape for zone '{self.name}': {e}")

    @property
    def area_m2(self) -> float:
        """Returns the real-world physical ground area in square meters."""
        if self.polygon is not None:
            return float(self.polygon.area)
        return 0.0

    @property
    def perimeter_length_m(self) -> float:
        """Returns the physical length in meters (perimeter or tripwire length)."""
        if self.line is not None:
            return float(self.line.length)
        elif self.polygon is not None:
            return float(self.polygon.length)
        return 0.0

    def contains_point(self, X_meters: float, Y_meters: float) -> bool:
        """Checks if a metric point (X, Y) lies inside this ground polygon."""
        if self.polygon is None:
            return False
        if not (np.isfinite(X_meters) and np.isfinite(Y_meters)):
            return False
        pt = Point(float(X_meters), float(Y_meters))
        try:
            return bool(self.polygon.contains(pt) or self.polygon.touches(pt))
        except Exception:
            return False

    def check_tripwire_crossing(
        self,
        prev_world_pt: Tuple[float, float],
        curr_world_pt: Tuple[float, float]
    ) -> Optional[str]:
        """
        Calculates directional tripwire crossing in metric ground space.
        Returns 'CROSS_IN', 'CROSS_OUT', or None.

        """
        if self.line is None or prev_world_pt is None or curr_world_pt is None:
            return None

        if len(prev_world_pt) < 2 or len(curr_world_pt) < 2:
            return None

        if not (np.isfinite(prev_world_pt[0]) and np.isfinite(prev_world_pt[1]) and
                np.isfinite(curr_world_pt[0]) and np.isfinite(curr_world_pt[1])):
            return None

        try:
            motion_seg = LineString([(float(prev_world_pt[0]), float(prev_world_pt[1])),
                                     (float(curr_world_pt[0]), float(curr_world_pt[1]))])
            if not self.line.intersects(motion_seg):
                return None
        except Exception:
            return None

        a = self.world_coords[0]
        b = self.world_coords[1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]

        def cross_product(p):
            return dx * (p[1] - a[1]) - dy * (p[0] - a[0])

        cp_prev = cross_product(prev_world_pt)
        cp_curr = cross_product(curr_world_pt)

        eps = 1e-5
        # Both strictly on opposite sides — normal crossing
        if cp_prev > eps and cp_curr < -eps:
            return "CROSS_IN"
        elif cp_prev < -eps and cp_curr > eps:
            return "CROSS_OUT"

        return None
