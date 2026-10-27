import pytest
from backend.app.core.zones import ZoneManager, ZoneGeometry

class MockZone:
    def __init__(self, id, name, zone_type, coords, armed=True):
        self.id = id
        self.name = name
        self.zone_type = zone_type
        self.coordinates = coords
        self.armed = armed
        self.schedule_start = None
        self.schedule_end = None
        self.color = "#EF4444"

def test_enter_debounce_and_exit_lifecycle():
    zm = ZoneManager()
    # Zone: square from [0.2, 0.2] to [0.8, 0.8]
    coords = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    zm.update_zones([MockZone("z1", "Secure Vault", "polygon", coords)])

    t = 100.0
    # Step 1: Track 1 starts outside at (0.1, 0.1)
    evs_1 = zm.process_tracks(
        tracks=[{"track_id": 1, "norm_foot": (0.1, 0.1), "confidence": 0.85}],
        current_time=t,
        debounce_frames=2
    )
    assert len(evs_1) == 0

    # Step 2: Track 1 moves inside to (0.5, 0.5) - Frame 1 inside
    t += 0.04
    evs_2 = zm.process_tracks(
        tracks=[{"track_id": 1, "norm_foot": (0.5, 0.5), "confidence": 0.88}],
        current_time=t,
        debounce_frames=2
    )
    # Debounce requires 2 frames inside, so frame 1 does not trigger ENTER
    assert len(evs_2) == 0

    # Step 3: Track 1 remains inside at (0.5, 0.5) - Frame 2 inside
    t += 0.04
    evs_3 = zm.process_tracks(
        tracks=[{"track_id": 1, "norm_foot": (0.5, 0.5), "confidence": 0.89}],
        current_time=t,
        debounce_frames=2
    )
    # Debounce met! Exactly 1 ENTER event
    assert len(evs_3) == 1
    assert evs_3[0]["event_type"] == "ENTER"
    assert evs_3[0]["track_id"] == 1
    assert evs_3[0]["zone_id"] == "z1"

    # Step 4: Track 1 stays inside for several frames (testing NO DUPLICATE SPAM)
    for _ in range(10):
        t += 0.04
        evs_cont = zm.process_tracks(
            tracks=[{"track_id": 1, "norm_foot": (0.52, 0.51), "confidence": 0.90}],
            current_time=t,
            debounce_frames=2,
            loiter_sec=10.0
        )
        assert len(evs_cont) == 0  # No duplicate ENTER alarms while inside!

    # Step 5: Track 1 moves outside to (0.15, 0.5)
    t += 0.04
    evs_exit = zm.process_tracks(
        tracks=[{"track_id": 1, "norm_foot": (0.15, 0.5), "confidence": 0.85}],
        current_time=t,
        debounce_frames=2
    )
    assert len(evs_exit) == 1
    assert evs_exit[0]["event_type"] == "EXIT"
    assert evs_exit[0]["track_id"] == 1

def test_loitering_alert():
    zm = ZoneManager()
    coords = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    zm.update_zones([MockZone("z1", "Restricted Area", "polygon", coords)])

    t = 200.0
    # Enter zone with debounce
    zm.process_tracks([{"track_id": 2, "norm_foot": (0.5, 0.5), "confidence": 0.9}], t, debounce_frames=1)

    # 4 seconds later (loiter threshold is 5s) -> No loiter event yet
    t += 4.0
    evs_early = zm.process_tracks([{"track_id": 2, "norm_foot": (0.5, 0.5), "confidence": 0.9}], t, loiter_sec=5.0)
    assert len(evs_early) == 0

    # 6 seconds total inside -> LOITER event must trigger
    t += 2.0
    evs_loiter = zm.process_tracks([{"track_id": 2, "norm_foot": (0.5, 0.5), "confidence": 0.9}], t, loiter_sec=5.0)
    assert len(evs_loiter) == 1
    assert evs_loiter[0]["event_type"] == "LOITER"
    assert evs_loiter[0]["track_id"] == 2

    # Further seconds inside -> LOITER should not spam repeatedly
    t += 2.0
    evs_loiter_repeat = zm.process_tracks([{"track_id": 2, "norm_foot": (0.5, 0.5), "confidence": 0.9}], t, loiter_sec=5.0)
    assert len(evs_loiter_repeat) == 0
