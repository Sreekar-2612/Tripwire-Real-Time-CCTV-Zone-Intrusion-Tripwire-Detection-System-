import pytest
from datetime import datetime
from backend.app.core.zones import ZoneGeometry

def test_polygon_containment():
    coords = [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]]
    zone = ZoneGeometry(
        zone_id="zone-1",
        name="Restricted Courtyard",
        zone_type="polygon",
        coordinates=coords,
        armed=True
    )

    # Point clearly inside
    assert zone.contains_point(0.4, 0.4) is True
    # Points clearly outside
    assert zone.contains_point(0.1, 0.1) is False
    assert zone.contains_point(0.7, 0.5) is False
    assert zone.contains_point(0.4, 0.9) is False

def test_tripwire_crossing_direction():
    # Vertical tripwire line at x=0.5 from y=0.1 to y=0.9
    coords = [[0.5, 0.1], [0.5, 0.9]]
    zone = ZoneGeometry(
        zone_id="line-1",
        name="Perimeter Tripwire",
        zone_type="tripwire",
        coordinates=coords,
        armed=True
    )

    # Cross from left (0.3) to right (0.7)
    res_in = zone.check_tripwire_crossing((0.3, 0.5), (0.7, 0.5))
    assert res_in in ["CROSS_IN", "CROSS_OUT"]

    # Cross reverse from right (0.7) to left (0.3)
    res_out = zone.check_tripwire_crossing((0.7, 0.5), (0.3, 0.5))
    assert res_out in ["CROSS_IN", "CROSS_OUT"]
    assert res_in != res_out  # Opposite directions!

    # No crossing (stays on left)
    res_none = zone.check_tripwire_crossing((0.2, 0.5), (0.4, 0.5))
    assert res_none is None

def test_zone_schedule():
    coords = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    zone = ZoneGeometry(
        zone_id="sched-1",
        name="Scheduled Zone",
        zone_type="polygon",
        coordinates=coords,
        armed=True,
        schedule_start="09:00",
        schedule_end="18:00"
    )

    # Within hours (14:30)
    dt_active = datetime(2026, 10, 27, 14, 30)
    assert zone.is_armed_at(dt_active) is True

    # Outside hours (20:15)
    dt_inactive = datetime(2026, 10, 27, 20, 15)
    assert zone.is_armed_at(dt_inactive) is False

    # Disarmed zone
    zone.armed = False
    assert zone.is_armed_at(dt_active) is False
