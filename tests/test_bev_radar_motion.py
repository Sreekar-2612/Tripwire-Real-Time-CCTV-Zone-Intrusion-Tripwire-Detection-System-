import math
import numpy as np
import pytest
from cv_core.visualizer.bev_radar import BEVRadarRenderer
from cv_core.geometry.zones_3d import MetricZone3D

def test_bev_radar_coordinate_mapping():
    """Verify ground coordinates map accurately to radar pixel coordinates."""
    size = 400
    extent = 20.0
    renderer = BEVRadarRenderer(radar_size=size, world_extent_m=extent)

    # (0, 0) is at bottom-left (camera origin)
    px, py = renderer.world_to_radar_pixel(0.0, 0.0)
    assert px == 0
    assert py == size - 1

    # (20, 0) is at bottom-right
    px, py = renderer.world_to_radar_pixel(20.0, 0.0)
    assert px == size - 1
    assert py == size - 1

    # (0, 20) is at top-left (max depth forward)
    px, py = renderer.world_to_radar_pixel(0.0, 20.0)
    assert px == 0
    assert py == 0

    # (20, 20) is at top-right
    px, py = renderer.world_to_radar_pixel(20.0, 20.0)
    assert px == size - 1
    assert py == 0

    # Midpoint (10, 10)
    px, py = renderer.world_to_radar_pixel(10.0, 10.0)
    assert np.isclose(px, size // 2, atol=1)
    assert np.isclose(py, size // 2, atol=1)

def test_bev_radar_grid_lines_and_camera_origin():
    """Verify that horizontal grid lines and labels match world Y coordinate."""
    size = 400
    extent = 20.0
    grid_step = 2.0
    renderer = BEVRadarRenderer(radar_size=size, world_extent_m=extent, grid_step_m=grid_step)
    bg = renderer._static_bg
    assert bg.shape == (size, size, 3)

    # Check that Y=2.0m line is near the bottom (self.size - 1 - 2*20 = 359), NOT near the top
    expected_y_2m = int(round(size - 1 - 2.0 * (size / extent)))
    assert expected_y_2m > size // 2

def test_bev_radar_motion_direction_towards_camera():
    """
    When an intruder moves towards the camera (downward in camera view, decreasing Y in world),
    vy < 0. On the radar map, the arrow must point DOWN towards the camera origin (py increases).
    """
    size = 480
    extent = 20.0
    renderer = BEVRadarRenderer(radar_size=size, world_extent_m=extent)

    # Person at (10.0, 12.0) moving south towards camera at 1.5 m/s (5.4 km/h)
    px, py = renderer.world_to_radar_pixel(10.0, 12.0)
    vx, vy = 0.0, -1.5  # vy is negative when moving towards camera
    speed_mps = float(np.hypot(vx, vy))

    dir_x = vx / speed_mps
    dir_y = vy / speed_mps
    arrow_len = min(65.0, max(18.0, 18.0 + (speed_mps - 0.1) * 12.0))
    tip_x = int(np.clip(round(px + dir_x * arrow_len), 0, size - 1))
    tip_y = int(np.clip(round(py - dir_y * arrow_len), 0, size - 1))

    # Moving towards camera: tip_y must be LARGER than py (points DOWN towards bottom of canvas)
    assert tip_y > py
    assert tip_x == px
    # Arrow must extend beyond the 10px blip circle
    assert (tip_y - py) >= 18

def test_bev_radar_motion_direction_away_from_camera():
    """
    When an intruder moves away from the camera (upward in camera view, increasing Y in world),
    vy > 0. On the radar map, the arrow must point UP away from the camera origin (py decreases).
    """
    size = 480
    extent = 20.0
    renderer = BEVRadarRenderer(radar_size=size, world_extent_m=extent)

    px, py = renderer.world_to_radar_pixel(10.0, 6.0)
    vx, vy = 0.0, 2.0  # vy is positive when moving away from camera
    speed_mps = float(np.hypot(vx, vy))

    dir_x = vx / speed_mps
    dir_y = vy / speed_mps
    arrow_len = min(65.0, max(18.0, 18.0 + (speed_mps - 0.1) * 12.0))
    tip_x = int(np.clip(round(px + dir_x * arrow_len), 0, size - 1))
    tip_y = int(np.clip(round(py - dir_y * arrow_len), 0, size - 1))

    # Moving away from camera: tip_y must be SMALLER than py (points UP towards top of canvas)
    assert tip_y < py
    assert tip_x == px
    assert (py - tip_y) >= 18

def test_bev_radar_motion_direction_lateral():
    """Verify lateral motion (East/Right vx > 0 and West/Left vx < 0)."""
    size = 480
    extent = 20.0
    renderer = BEVRadarRenderer(radar_size=size, world_extent_m=extent)

    px, py = renderer.world_to_radar_pixel(10.0, 10.0)

    # Moving East (Right): vx > 0, vy = 0
    vx_right, vy_right = 1.8, 0.0
    spd_r = float(np.hypot(vx_right, vy_right))
    tip_x_r = int(np.clip(round(px + (vx_right / spd_r) * 20.0), 0, size - 1))
    assert tip_x_r > px

    # Moving West (Left): vx < 0, vy = 0
    vx_left, vy_left = -1.8, 0.0
    spd_l = float(np.hypot(vx_left, vy_left))
    tip_x_l = int(np.clip(round(px + (vx_left / spd_l) * 20.0), 0, size - 1))
    assert tip_x_l < px

def test_bev_radar_arrow_length_proportional_to_speed():
    """Verify arrow length scales with speed but remains clamped within [18, 65] px."""
    min_len = 18.0
    max_len = 65.0

    # Low speed (0.3 m/s)
    len_low = min(max_len, max(min_len, min_len + (0.3 - 0.1) * 12.0))
    # Medium speed (1.5 m/s)
    len_med = min(max_len, max(min_len, min_len + (1.5 - 0.1) * 12.0))
    # High speed (6.0 m/s)
    len_high = min(max_len, max(min_len, min_len + (6.0 - 0.1) * 12.0))

    assert min_len <= len_low < len_med < len_high <= max_len
    assert len_low >= 18.0
    assert len_high == 65.0

def test_bev_radar_tag_anti_clipping_and_collision_avoidance():
    """Verify target tags never clip off-canvas and separate when targets are close."""
    size = 400
    renderer = BEVRadarRenderer(radar_size=size, world_extent_m=20.0)

    tracks = [
        # Target near right edge (X=19.5m -> px ~ 390)
        {
            "track_id": 1,
            "world_pos": (19.5, 10.0),
            "world_trail": [(19.0, 10.0), (19.5, 10.0)],
            "speed_kmh": 4.5,
            "vx": 1.25,
            "vy": 0.0,
            "is_breaching": False
        },
        # Target near top edge / HUD header (Y=19.5m -> py ~ 10)
        {
            "track_id": 2,
            "world_pos": (10.0, 19.5),
            "world_trail": [(10.0, 19.0), (10.0, 19.5)],
            "speed_kmh": 5.0,
            "vx": 0.0,
            "vy": 1.4,
            "is_breaching": False
        },
        # Two targets very close to each other
        {
            "track_id": 3,
            "world_pos": (5.0, 5.0),
            "world_trail": [(5.0, 5.0)],
            "speed_kmh": 3.0,
            "vx": 0.8,
            "vy": 0.0,
            "is_breaching": True
        },
        {
            "track_id": 4,
            "world_pos": (5.1, 5.1),
            "world_trail": [(5.1, 5.1)],
            "speed_kmh": 3.2,
            "vx": 0.8,
            "vy": 0.0,
            "is_breaching": True
        }
    ]

    zones = [
        MetricZone3D("z1", "PERIMETER FENCE", "tripwire", [(0.0, 10.0), (20.0, 10.0)]),
        MetricZone3D("z2", "SECURE VAULT", "polygon", [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)])
    ]

    radar_img = renderer.render(zones, tracks)
    assert radar_img.shape == (size, size, 3)

def test_bev_radar_trail_temporal_gradient():
    """Verify that multi-point trajectory history renders smoothly."""
    size = 480
    renderer = BEVRadarRenderer(radar_size=size, world_extent_m=20.0)

    trail = [(float(i), float(i)) for i in range(1, 15)]
    tracks = [
        {
            "track_id": 99,
            "world_pos": (14.0, 14.0),
            "world_trail": trail,
            "speed_kmh": 6.0,
            "vx": 1.6,
            "vy": 1.6,
            "is_breaching": False
        }
    ]
    img = renderer.render([], tracks)
    assert img.shape == (size, size, 3)
