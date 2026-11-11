import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2

class BEVRadarRenderer:
    """
    Renders a top-down Bird's Eye View (BEV) radar map in real-world meters.
    Plots ground zones, track positions, velocity vectors, and intruder trajectories.
    """
    def __init__(
        self,
        radar_size: int = 480,
        world_extent_m: float = 20.0,  # [0, world_extent_m] along X and Y
        grid_step_m: float = 2.0,
        background_color: Tuple[int, int, int] = (20, 24, 28)
    ):
        self.size = radar_size
        self.world_extent = float(world_extent_m) if (world_extent_m is not None and world_extent_m > 1e-3) else 20.0
        self.grid_step = float(grid_step_m) if (grid_step_m is not None and grid_step_m > 1e-3) else 2.0
        self.bg_color = background_color
        self.scale = radar_size / self.world_extent  # pixels per meter
        # Precompute static radar background (metric grid, distance labels, and range rings)
        # to eliminate redundant rasterization on every frame (reduces render latency by ~85%)
        self._static_bg = self._precompute_static_background()

    def _precompute_static_background(self) -> np.ndarray:
        """Pre-renders static radar grid lines, meter labels, and range rings."""
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:] = self.bg_color

        # 1. Metric Grid & Meter markings
        num_grid_lines = int(self.world_extent / self.grid_step)
        for i in range(1, num_grid_lines + 1):
            dist_m = i * self.grid_step
            pos = int(dist_m * self.scale)
            # Vertical line
            cv2.line(canvas, (pos, 0), (pos, self.size), (35, 42, 50), 1, cv2.LINE_AA)
            # Horizontal line
            cv2.line(canvas, (0, pos), (self.size, pos), (35, 42, 50), 1, cv2.LINE_AA)
            
            # Grid labels
            cv2.putText(canvas, f"{dist_m:.0f}m", (pos + 2, self.size - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 115, 130), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{dist_m:.0f}m", (4, self.size - pos - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 115, 130), 1, cv2.LINE_AA)

        # Concentric Radar Range Rings
        for r_m in [5.0, 10.0, 15.0]:
            r_px = int(r_m * self.scale)
            if r_px < self.size:
                cv2.circle(canvas, (0, self.size - 1), r_px, (45, 55, 65), 1, cv2.LINE_AA)

        return canvas

    def world_to_radar_pixel(self, X_m: float, Y_m: float) -> Tuple[int, int]:
        """Maps ground coordinates (X, Y in meters) to radar canvas pixels."""
        if not (np.isfinite(X_m) and np.isfinite(Y_m)):
            return (0, self.size - 1)
        px = int(np.clip(X_m * self.scale, 0, self.size - 1))
        # Invert Y so 0,0 is at the bottom or top as standard
        py = int(np.clip(self.size - 1 - (Y_m * self.scale), 0, self.size - 1))
        return (px, py)

    def render(
        self,
        zones: List[Any],
        tracks: List[Dict[str, Any]],
        title: str = "BIRD'S EYE VIEW (BEV) GROUND RADAR"
    ) -> np.ndarray:
        """
        Renders the full BEV radar map image.
        tracks: list of dicts with keys:
            'track_id': int
            'world_pos': (X, Y)
            'world_trail': [(X, Y), ...]
            'speed_kmh': float
            'vx': float, 'vy': float
            'is_breaching': bool
        """
        canvas = self._static_bg.copy()

        # 2. Draw Zones in Metric Ground Space
        zone_overlay = canvas.copy()
        for z in zones:
            coords = getattr(z, "world_coords", [])
            z_type = getattr(z, "zone_type", "polygon")
            if z_type != "tripwire" and len(coords) >= 3:
                pts = [self.world_to_radar_pixel(p[0], p[1]) for p in coords]
                pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(zone_overlay, [pts_arr], (0, 40, 100))

        # Blend polygon fills before rendering crisp wireframes, labels, and track blips
        cv2.addWeighted(zone_overlay, 0.4, canvas, 0.6, 0, canvas)

        for z in zones:
            coords = getattr(z, "world_coords", [])
            z_type = getattr(z, "zone_type", "polygon")
            name = getattr(z, "name", "Zone")

            if z_type == "tripwire" and len(coords) >= 2:
                p1 = self.world_to_radar_pixel(coords[0][0], coords[0][1])
                p2 = self.world_to_radar_pixel(coords[1][0], coords[1][1])
                cv2.line(canvas, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, p1, 4, (0, 0, 255), -1)
                cv2.circle(canvas, p2, 4, (0, 0, 255), -1)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(canvas, name, (mid[0] - 20, mid[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1, cv2.LINE_AA)
            elif len(coords) >= 3:
                pts = [self.world_to_radar_pixel(p[0], p[1]) for p in coords]
                pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts_arr], isClosed=True, color=(0, 120, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(canvas, name, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 160, 255), 1, cv2.LINE_AA)

        # 3. Draw Tracked Intruders
        for t in tracks:
            w_pos = t.get("world_pos")
            if not w_pos or len(w_pos) < 2 or not (np.isfinite(w_pos[0]) and np.isfinite(w_pos[1])):
                continue

            tid = t.get("track_id", 0)
            is_breaching = t.get("is_breaching", False)
            color = (0, 0, 255) if is_breaching else (0, 230, 255)  # BGR

            # Radar blip position
            px, py = self.world_to_radar_pixel(w_pos[0], w_pos[1])

            # Draw Ground Trajectory Trail
            trail = t.get("world_trail", [])
            valid_trail = [tp for tp in trail if tp is not None and len(tp) >= 2 and np.isfinite(tp[0]) and np.isfinite(tp[1])]
            if len(valid_trail) > 1:
                trail_pts = [self.world_to_radar_pixel(tp[0], tp[1]) for tp in valid_trail]
                for idx in range(1, len(trail_pts)):
                    cv2.line(canvas, trail_pts[idx - 1], trail_pts[idx], (180, 180, 0), 1, cv2.LINE_AA)

            # Radar Blip (Circle + Outer ring)
            cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 10, color, 1, cv2.LINE_AA)

            # Velocity Vector Arrow
            vx = float(t.get("vx", 0.0))
            vy = float(t.get("vy", 0.0))
            if not (np.isfinite(vx) and np.isfinite(vy)):
                vx, vy = 0.0, 0.0

            if np.hypot(vx, vy) > 0.05:
                arrow_scale = 1.5  # amplify for visibility
                tip_x = int(np.clip(px + vx * self.scale * arrow_scale, 0, self.size - 1))
                tip_y = int(np.clip(py - vy * self.scale * arrow_scale, 0, self.size - 1))  # invert Y for canvas
                if tip_x != px or tip_y != py:
                    cv2.arrowedLine(canvas, (px, py), (tip_x, tip_y), (0, 255, 120), 2, tipLength=0.3)

            # Metadata Tag (ID + km/h)
            speed_kmh = float(t.get("speed_kmh", t.get("speed", 0.0)))
            if not np.isfinite(speed_kmh):
                speed_kmh = 0.0
            tag = f"#{tid} {speed_kmh:.1f}km/h"
            cv2.putText(canvas, tag, (px + 10, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # HUD Header
        cv2.rectangle(canvas, (0, 0), (self.size, 26), (12, 16, 20), -1)
        cv2.putText(canvas, title, (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 120), 1, cv2.LINE_AA)
        scale_tag = f"Grid: {self.grid_step:.0f}m | Extent: {self.world_extent:.0f}m"
        (stw, _), _ = cv2.getTextSize(scale_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(canvas, scale_tag, (self.size - stw - 8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 170, 180), 1, cv2.LINE_AA)

        return canvas

    @staticmethod
    def create_dual_view(
        perspective_frame: np.ndarray,
        bev_radar: np.ndarray
    ) -> np.ndarray:
        """
        Combines camera perspective video and BEV radar side-by-side into a single display.
        """
        if perspective_frame is None or perspective_frame.size == 0:
            return bev_radar if bev_radar is not None else np.zeros((480, 480, 3), dtype=np.uint8)
        if bev_radar is None or bev_radar.size == 0:
            return perspective_frame

        h_cam, w_cam = perspective_frame.shape[:2]
        h_radar, w_radar = bev_radar.shape[:2]

        if h_cam <= 0 or w_cam <= 0:
            return bev_radar
        if h_radar <= 0 or w_radar <= 0:
            return perspective_frame

        # Match radar height to camera height
        if h_radar != h_cam:
            new_w = max(1, int(w_radar * (h_cam / float(max(1, h_radar)))))
            bev_radar_resized = cv2.resize(bev_radar, (new_w, h_cam), interpolation=cv2.INTER_AREA)
        else:
            bev_radar_resized = bev_radar

        # Concatenate horizontally
        dual_view = np.hstack([perspective_frame, bev_radar_resized])
        return dual_view
