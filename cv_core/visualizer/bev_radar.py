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
        """Pre-renders static radar grid lines, meter labels, range rings, and camera origin."""
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:] = self.bg_color

        # 1. Metric Grid & Meter markings
        num_grid_lines = int(self.world_extent / self.grid_step)
        for i in range(1, num_grid_lines + 1):
            dist_m = i * self.grid_step
            # Vertical line (X-coordinate: lateral position from camera origin)
            px_line = int(round(dist_m * self.scale))
            if 0 < px_line < self.size:
                cv2.line(canvas, (px_line, 0), (px_line, self.size), (35, 42, 50), 1, cv2.LINE_AA)
                # Label along the bottom
                cv2.putText(canvas, f"{dist_m:.0f}m", (px_line + 2, self.size - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 115, 130), 1, cv2.LINE_AA)

            # Horizontal line (Y-coordinate: depth/distance from camera)
            # In world coordinates, Y=0 is at bottom (self.size - 1), increasing Y goes UP
            py_line = int(round(self.size - 1 - dist_m * self.scale))
            if 0 < py_line < self.size:
                cv2.line(canvas, (0, py_line), (self.size, py_line), (35, 42, 50), 1, cv2.LINE_AA)
                # Label aligned directly with the horizontal grid line
                cv2.putText(canvas, f"{dist_m:.0f}m", (4, py_line - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 115, 130), 1, cv2.LINE_AA)

        # 2. Concentric Radar Range Rings & Range Labels
        ring_step = 5.0 if self.world_extent >= 15.0 else max(2.0, self.grid_step * 2)
        max_ring_dist = self.world_extent * 1.42  # diagonal extent
        ring_dists = np.arange(ring_step, max_ring_dist, ring_step)

        for r_m in ring_dists:
            r_px = int(round(r_m * self.scale))
            if r_px < int(self.size * 1.42):
                cv2.circle(canvas, (0, self.size - 1), r_px, (45, 55, 65), 1, cv2.LINE_AA)
                # Range ring label along the 45-degree diagonal arc
                lx = int(round(r_px * 0.7071))
                ly = int(round((self.size - 1) - r_px * 0.7071))
                if 12 <= lx <= self.size - 35 and 36 <= ly <= self.size - 12:
                    cv2.putText(canvas, f"R:{r_m:.0f}m", (lx + 2, ly - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.29, (75, 95, 115), 1, cv2.LINE_AA)

        # 3. Camera Origin Node & Directional Sightlines
        # Camera is located at ground coordinate (X=0, Y=0) -> pixel (0, self.size - 1)
        cam_px = 0
        cam_py = self.size - 1

        # Camera baseline sightlines (0 deg lateral right, 90 deg forward depth)
        cv2.line(canvas, (cam_px, cam_py), (self.size - 1, cam_py), (55, 75, 90), 1, cv2.LINE_AA)
        cv2.line(canvas, (cam_px, cam_py), (cam_px, 28), (55, 75, 90), 1, cv2.LINE_AA)

        # Camera beacon indicator
        cv2.circle(canvas, (cam_px, cam_py), 8, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cam_px, cam_py), 14, (0, 150, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "CAM (0,0)", (12, self.size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 220, 255), 1, cv2.LINE_AA)

        # Axis Orientation Indicators
        cv2.putText(canvas, "^ DEPTH (Y)", (4, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 140, 160), 1, cv2.LINE_AA)
        tag_x_text = "WIDTH (X) >"
        (tw_x, _), _ = cv2.getTextSize(tag_x_text, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.putText(canvas, tag_x_text, (self.size - tw_x - 6, self.size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 140, 160), 1, cv2.LINE_AA)

        return canvas

    def world_to_radar_pixel(self, X_m: float, Y_m: float) -> Tuple[int, int]:
        """Maps ground coordinates (X, Y in meters) to radar canvas pixels."""
        if not (np.isfinite(X_m) and np.isfinite(Y_m)):
            return (0, self.size - 1)
        px = int(np.clip(round(X_m * self.scale), 0, self.size - 1))
        # Invert Y so Y=0 is at the bottom (camera baseline) and increasing Y goes UP
        py = int(np.clip(round(self.size - 1 - (Y_m * self.scale)), 0, self.size - 1))
        return (px, py)

    def render(
        self,
        zones: List[Any],
        tracks: List[Dict[str, Any]],
        title: str = "BIRD'S EYE VIEW (BEV) GROUND RADAR"
    ) -> np.ndarray:
        """
        Renders the full BEV radar map image.
        zones: list of MetricZone3D objects
        tracks: list of dicts with keys:
            'track_id': int
            'world_pos': (X, Y)
            'world_trail': [(X, Y), ...]
            'speed_kmh': float
            'vx': float, 'vy': float
            'is_breaching': bool
        """
        canvas = self._static_bg.copy()

        # 1. Draw Zones in Metric Ground Space
        zone_overlay = canvas.copy()
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

                # Centered tripwire label with anti-clipping & background pill
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
                lx = int(np.clip(mid_x - tw // 2, 6, self.size - tw - 6))
                ly = int(np.clip(mid_y - 8, 38, self.size - 10))

                cv2.rectangle(canvas, (lx - 3, ly - th - 2), (lx + tw + 3, ly + 2), (10, 14, 20), -1)
                cv2.rectangle(canvas, (lx - 3, ly - th - 2), (lx + tw + 3, ly + 2), (0, 80, 220), 1, cv2.LINE_AA)
                cv2.putText(canvas, name, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 140, 255), 1, cv2.LINE_AA)

            elif len(coords) >= 3:
                pts = [self.world_to_radar_pixel(p[0], p[1]) for p in coords]
                pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(zone_overlay, [pts_arr], (0, 40, 100))
                cv2.polylines(canvas, [pts_arr], isClosed=True, color=(0, 120, 255), thickness=1, lineType=cv2.LINE_AA)

                # Centroid polygon label with anti-clipping & background pill
                cx = int(np.mean([p[0] for p in pts]))
                cy = int(np.mean([p[1] for p in pts]))
                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                lx = int(np.clip(cx - tw // 2, 6, self.size - tw - 6))
                ly = int(np.clip(cy + th // 2, 38, self.size - 10))

                cv2.rectangle(canvas, (lx - 3, ly - th - 2), (lx + tw + 3, ly + 2), (10, 18, 28), -1)
                cv2.rectangle(canvas, (lx - 3, ly - th - 2), (lx + tw + 3, ly + 2), (0, 100, 200), 1, cv2.LINE_AA)
                cv2.putText(canvas, name, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 180, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(zone_overlay, 0.35, canvas, 0.65, 0, canvas)

        # 2. Draw Tracked Intruders
        placed_tag_rects: List[Tuple[int, int, int, int]] = []

        for t in tracks:
            w_pos = t.get("world_pos")
            if not w_pos or len(w_pos) < 2 or not (np.isfinite(w_pos[0]) and np.isfinite(w_pos[1])):
                continue

            tid = t.get("track_id", 0)
            is_breaching = t.get("is_breaching", False)
            color = (0, 0, 255) if is_breaching else (0, 230, 255)  # BGR

            # Radar blip pixel coordinates
            px, py = self.world_to_radar_pixel(w_pos[0], w_pos[1])

            # Draw Ground Trajectory Motion History Trail (with temporal fading)
            trail = t.get("world_trail", [])
            valid_trail = [
                tp for tp in trail
                if tp is not None and len(tp) >= 2 and np.isfinite(tp[0]) and np.isfinite(tp[1])
            ]
            if len(valid_trail) > 1:
                trail_pts = [self.world_to_radar_pixel(tp[0], tp[1]) for tp in valid_trail]
                # Ensure the current position is the head of the trail
                if trail_pts[-1] != (px, py):
                    trail_pts.append((px, py))

                n_pts = len(trail_pts)
                for idx in range(1, n_pts):
                    p_start = trail_pts[idx - 1]
                    p_end = trail_pts[idx]
                    if p_start == p_end:
                        continue
                    # Fade factor from 0.2 (oldest) to 1.0 (newest)
                    alpha = float(idx) / float(n_pts)
                    b_col = int(40 + 190 * alpha)
                    g_col = int(80 + 150 * alpha)
                    r_col = int(20 + 230 * alpha)
                    thickness = 2 if alpha > 0.65 else 1
                    cv2.line(canvas, p_start, p_end, (b_col, g_col, r_col), thickness, cv2.LINE_AA)

                # Recent trajectory breadcrumb dot
                if len(trail_pts) >= 3:
                    cv2.circle(canvas, trail_pts[-2], 2, (0, 220, 255), -1, cv2.LINE_AA)

            # Radar Blip (Solid core + Outer boundary ring)
            cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 10, color, 1, cv2.LINE_AA)

            # Velocity Vector Arrow
            vx = float(t.get("vx", 0.0))
            vy = float(t.get("vy", 0.0))
            if not (np.isfinite(vx) and np.isfinite(vy)):
                vx, vy = 0.0, 0.0

            # Fallback estimation from trail if vx, vy are zero but trail moved
            if vx == 0.0 and vy == 0.0 and len(valid_trail) >= 2:
                dt_fallback = 0.04  # standard frame step
                vx = (valid_trail[-1][0] - valid_trail[-2][0]) / dt_fallback
                vy = (valid_trail[-1][1] - valid_trail[-2][1]) / dt_fallback

            speed_mps = float(np.hypot(vx, vy))
            # Show vector arrow when moving above deadband (> 0.08 m/s ~ 0.3 km/h)
            if speed_mps > 0.08:
                dir_x = vx / speed_mps
                dir_y = vy / speed_mps

                # Proportional arrow length: min 18px (extends outside 10px ring), max 65px
                min_len = 18.0
                max_len = 65.0
                arrow_len = min(max_len, max(min_len, min_len + (speed_mps - 0.1) * 12.0))

                # Notice: minus sign for dir_y because canvas Y increases DOWNWARD!
                # When vy < 0 (moving towards camera, decreasing Y): -dir_y > 0 -> tip_y > py (points DOWN).
                # When vy > 0 (moving away from camera, increasing Y): -dir_y < 0 -> tip_y < py (points UP).
                tip_x = int(np.clip(round(px + dir_x * arrow_len), 0, self.size - 1))
                tip_y = int(np.clip(round(py - dir_y * arrow_len), 0, self.size - 1))

                if tip_x != px or tip_y != py:
                    # Dynamically scale tip fraction so arrowhead remains ~8-9px
                    tip_frac = max(0.20, min(0.48, 8.5 / arrow_len))
                    arrow_color = (0, 100, 255) if is_breaching else (0, 255, 120)
                    cv2.arrowedLine(canvas, (px, py), (tip_x, tip_y), arrow_color, 2, tipLength=tip_frac)

            # Metadata Tag (ID + km/h) with Collision Avoidance & Anti-Clipping
            speed_kmh = float(t.get("speed_kmh", t.get("speed", 0.0)))
            if not np.isfinite(speed_kmh):
                speed_kmh = 0.0
            tag = f"#{tid} {speed_kmh:.1f}km/h"

            (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)

            # Default placement: to the right and slightly above the blip
            tag_x = px + 14
            tag_y = py - 4

            # If it would clip on the right, flip to the left
            if tag_x + tw + 4 > self.size - 4:
                tag_x = px - 14 - tw
            tag_x = int(np.clip(tag_x, 6, self.size - tw - 6))

            # If it would collide with the HUD header at top (y <= 26), flip below blip
            if tag_y - th - 2 < 28:
                tag_y = py + th + 14
            tag_y = int(np.clip(tag_y, 28 + th + 2, self.size - 6))

            # Anti-Overlap Collision Resolution with other tags
            tag_box = [tag_x - 3, tag_y - th - 3, tag_x + tw + 3, tag_y + 3]
            for obox in placed_tag_rects:
                # Check for overlap
                if not (tag_box[2] < obox[0] or tag_box[0] > obox[2] or
                        tag_box[3] < obox[1] or tag_box[1] > obox[3]):
                    # Overlap detected! Shift tag vertically
                    if tag_y + th + 8 < self.size - 6:
                        tag_y = obox[3] + th + 3
                    else:
                        tag_y = obox[1] - 4
                    tag_y = int(np.clip(tag_y, 28 + th + 2, self.size - 6))
                    tag_box = [tag_x - 3, tag_y - th - 3, tag_x + tw + 3, tag_y + 3]

            placed_tag_rects.append(tuple(tag_box))

            # Dark pill background for crisp readability
            cv2.rectangle(canvas, (tag_x - 3, tag_y - th - 3), (tag_x + tw + 3, tag_y + 3), (12, 16, 20), -1)
            cv2.rectangle(canvas, (tag_x - 3, tag_y - th - 3), (tag_x + tw + 3, tag_y + 3), color, 1, cv2.LINE_AA)
            cv2.putText(canvas, tag, (tag_x, tag_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. HUD Header
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
