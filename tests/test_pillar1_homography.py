import pytest
import numpy as np
import tempfile
from pathlib import Path

from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D
from cv_core.visualizer.bev_radar import BEVRadarRenderer

def test_homography_exact_correspondence_and_roundtrip():
    """Verify DLT homography calculation and sub-millimeter round-trip reprojection."""
    # Define an idealized trapezoid perspective mapping to a 4m x 8m rectangle
    img_pts = [
        (300.0, 150.0),  # Top-left (far)
        (500.0, 150.0),  # Top-right (far)
        (650.0, 500.0),  # Bottom-right (near)
        (150.0, 500.0)   # Bottom-left (near)
    ]
    world_pts = [
        (0.0, 8.0),
        (4.0, 8.0),
        (4.0, 0.0),
        (0.0, 0.0)
    ]

    transformer = HomographyTransformer.from_point_correspondences(img_pts, world_pts)
    assert transformer.is_calibrated

    # Check forward projection matches ground truth
    error = transformer.compute_reprojection_error(img_pts, world_pts)
    assert error < 1e-3, f"Reprojection error too high: {error}"

    # Check round-trip: (u, v) -> (X, Y) -> (u, v)
    for u_orig, v_orig in img_pts:
        X, Y = transformer.image_to_world(u_orig, v_orig)
        u_rec, v_rec = transformer.world_to_image(X, Y)
        assert np.isclose(u_orig, u_rec, atol=1e-3)
        assert np.isclose(v_orig, v_rec, atol=1e-3)

def test_homography_singular_matrix_validation():
    """Ensure singular or non-invertible matrix inputs raise ValueError."""
    singular_H = np.zeros((3, 3))
    with pytest.raises(ValueError):
        HomographyTransformer(H_matrix=singular_H)

def test_homography_velocity_and_speed_calculation():
    """Verify ground-plane speed calculation (m/s and km/h) and direction."""
    transformer = HomographyTransformer()
    # A person moves 3 meters east in 1.0 second: velocity = 3.0 m/s = 10.8 km/h, direction = 0 deg
    trajectory = [
        (1.0, 2.0, 100.0),
        (4.0, 2.0, 101.0)
    ]
    vel = transformer.compute_velocity(trajectory)
    assert np.isclose(vel["speed_mps"], 3.0, atol=1e-2)
    assert np.isclose(vel["speed_kmh"], 10.8, atol=1e-2)
    assert np.isclose(vel["vx"], 3.0, atol=1e-2)
    assert np.isclose(vel["vy"], 0.0, atol=1e-2)
    assert np.isclose(vel["direction_deg"], 0.0, atol=1e-1)

def test_metric_zone_3d_polygon_and_area():
    """Test metric ground polygon point containment and physical area calculation."""
    # A 5m x 4m rectangle zone: area = 20 m^2
    coords = [(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)]
    zone = MetricZone3D(
        zone_id="test_zone",
        name="Loading Bay",
        zone_type="polygon",
        world_coordinates=coords
    )

    assert np.isclose(zone.area_m2, 20.0, atol=1e-3)
    assert np.isclose(zone.perimeter_length_m, 18.0, atol=1e-3)

    # Center is inside
    assert zone.contains_point(2.5, 2.0)
    # Point outside
    assert not zone.contains_point(6.0, 2.0)

def test_metric_tripwire_directional_crossing():
    """Test crossing detection of a ground-plane tripwire vector."""
    # Tripwire along X-axis from (0, 5) to (10, 5)
    coords = [(0.0, 5.0), (10.0, 5.0)]
    wire = MetricZone3D(
        zone_id="tw_1",
        name="Perimeter Fence",
        zone_type="tripwire",
        world_coordinates=coords
    )

    # Moving from (5, 4) to (5, 6) crosses line
    cross_dir1 = wire.check_tripwire_crossing((5.0, 4.0), (5.0, 6.0))
    assert cross_dir1 in ["CROSS_IN", "CROSS_OUT"]

    # Moving reverse crosses in opposite direction
    cross_dir2 = wire.check_tripwire_crossing((5.0, 6.0), (5.0, 4.0))
    assert cross_dir2 in ["CROSS_IN", "CROSS_OUT"]
    assert cross_dir1 != cross_dir2

    # Moving parallel without intersection returns None
    assert wire.check_tripwire_crossing((2.0, 2.0), (4.0, 2.0)) is None

def test_bev_radar_renderer_and_dual_view():
    """Verify radar canvas generation and dual-view image compositing."""
    renderer = BEVRadarRenderer(radar_size=400, world_extent_m=20.0)
    
    zones = [
        MetricZone3D("z1", "Secure Area", "polygon", [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)])
    ]
    tracks = [
        {
            "track_id": 101,
            "world_pos": (5.0, 5.0),
            "world_trail": [(4.0, 4.0), (4.5, 4.5), (5.0, 5.0)],
            "speed_kmh": 4.5,
            "vx": 1.2,
            "vy": 1.2,
            "is_breaching": True
        }
    ]

    radar_img = renderer.render(zones, tracks)
    assert radar_img.shape == (400, 400, 3)

    # Test dual-view compositing with dummy camera frame (720x1280)
    camera_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dual_view = BEVRadarRenderer.create_dual_view(camera_frame, radar_img)
    assert dual_view.shape[0] == 720
    assert dual_view.shape[1] > 1280  # Combined width

def test_homography_json_serialization_roundtrip():
    """Verify saving and loading calibration data to disk."""
    H_orig = np.array([
        [1.2, 0.1, -50.0],
        [0.05, 0.9, -20.0],
        [0.0001, 0.0002, 1.0]
    ])
    transformer = HomographyTransformer(H_matrix=H_orig, world_bounds=(0.0, 0.0, 30.0, 30.0))
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        transformer.save_calibration(tmp_path)
        loaded = HomographyTransformer.load_calibration(tmp_path)
        assert np.allclose(transformer.H, loaded.H)
        assert transformer.world_bounds == loaded.world_bounds
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def test_homography_projective_singularity():
    """Verify that points mapping to infinity (zero denominator) raise ZeroDivisionError."""
    # Construct an H matrix where for (10, 0), denominator = 0 * 10 + 0 * 0 + 0 = 0
    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, -10.0]  # At u=10, v=0 -> denom = 10 - 10 = 0
    ])
    transformer = HomographyTransformer(H_matrix=H)
    with pytest.raises(ZeroDivisionError):
        transformer.image_to_world(10.0, 0.0)

def test_homography_reprojection_error_zero_for_identity():
    """Verify reprojection error calculation on identity transformation."""
    H = np.eye(3, dtype=np.float64)
    transformer = HomographyTransformer(H_matrix=H)
    pts = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    err = transformer.compute_reprojection_error(pts, pts)
    assert np.isclose(err, 0.0, atol=1e-6)

def test_metric_zone_tripwire_no_crossing_for_collinear_or_disjoint():
    """Verify that non-crossing or parallel segments return None."""
    coords = [(0.0, 5.0), (10.0, 5.0)]
    wire = MetricZone3D("tw_collinear", "Tripwire", "tripwire", coords)
    # Movement entirely on one side
    assert wire.check_tripwire_crossing((1.0, 2.0), (9.0, 2.0)) is None
    # Movement collinear with line
    assert wire.check_tripwire_crossing((1.0, 5.0), (8.0, 5.0)) is None

def test_homography_length_mismatch_raises_value_error():
    """Verify that unequal number of image and world correspondences raises ValueError."""
    img_pts = [(0, 0), (100, 0), (100, 100), (0, 100), (50, 50)]
    world_pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
    with pytest.raises(ValueError, match="must match"):
        HomographyTransformer.from_point_correspondences(img_pts, world_pts)

def test_project_ground_grid_handles_zero_or_negative_step():
    """Verify ground grid projection safely handles zero or negative step intervals."""
    transformer = HomographyTransformer(H_matrix=np.eye(3))
    # Test step = 0.0
    segs_zero = transformer.project_ground_grid_to_image((480, 640), grid_step_m=0.0)
    assert isinstance(segs_zero, list)
    # Test step = -2.0
    segs_neg = transformer.project_ground_grid_to_image((480, 640), grid_step_m=-2.0)
    assert isinstance(segs_neg, list)

