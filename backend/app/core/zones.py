import json
import logging
from datetime import datetime, time
from typing import List, Tuple, Dict, Optional, Any
from shapely.geometry import Polygon, Point, LineString
from shapely.validation import make_valid

logger = logging.getLogger("tripwire.zones")

class ZoneGeometry:
    """Handles spatial geometry tests for a single zone or tripwire."""
    def __init__(
        self,
        zone_id: str,
        name: str,
        zone_type: str,
        coordinates: List[List[float]],
        armed: bool = True,
        schedule_start: Optional[str] = None,
        schedule_end: Optional[str] = None,
        color: str = "#EF4444"
    ):
        self.zone_id = zone_id
        self.name = name
        self.zone_type = zone_type.lower()
        self.raw_coords = coordinates
        self.armed = armed
        self.schedule_start = schedule_start
        self.schedule_end = schedule_end
        self.color = color

        # Shapely shape
        self.polygon: Optional[Polygon] = None
        self.line: Optional[LineString] = None

        self._build_shape()

    def _build_shape(self):
        try:
            if self.zone_type == "tripwire":
                if len(self.raw_coords) >= 2:
                    self.line = LineString([(p[0], p[1]) for p in self.raw_coords[:2]])
            else:
                # Polygon zone
                if len(self.raw_coords) >= 3:
                    poly = Polygon([(p[0], p[1]) for p in self.raw_coords])
                    if not poly.is_valid:
                        poly = make_valid(poly)
                    self.polygon = poly
        except Exception as e:
            logger.error(f"Failed to build shape for zone {self.zone_id} ({self.name}): {e}")

    def is_armed_at(self, check_time: Optional[datetime] = None) -> bool:
        if not self.armed:
            return False
        
        if not self.schedule_start or not self.schedule_end:
            return True

        dt = check_time or datetime.now()
        cur_t = dt.time()

        try:
            s_hour, s_min = map(int, self.schedule_start.split(":"))
            e_hour, e_min = map(int, self.schedule_end.split(":"))
            start_t = time(s_hour, s_min)
            end_t = time(e_hour, e_min)

            if start_t <= end_t:
                return start_t <= cur_t <= end_t
            else:
                # Crosses midnight (e.g. 22:00 -> 06:00)
                return cur_t >= start_t or cur_t <= end_t
        except Exception as e:
            logger.warning(f"Error parsing schedule for zone {self.zone_id}: {e}")
            return True

    def contains_point(self, norm_x: float, norm_y: float) -> bool:
        """Test if normalized point (0.0 to 1.0) is inside the polygon."""
        if self.polygon is None:
            return False
        pt = Point(norm_x, norm_y)
        try:
            return self.polygon.contains(pt) or self.polygon.touches(pt)
        except Exception:
            return False

    def check_tripwire_crossing(
        self,
        prev_pt: Tuple[float, float],
        curr_pt: Tuple[float, float]
    ) -> Optional[str]:
        """
        Check if moving from prev_pt to curr_pt crossed this tripwire line.
        Returns: 'CROSS_IN', 'CROSS_OUT', or None.
        """
        if self.line is None or prev_pt is None or curr_pt is None:
            return None

        motion_seg = LineString([prev_pt, curr_pt])
        if not self.line.intersects(motion_seg):
            return None

        # Cross product to determine side and direction
        # Tripwire vector from A to B:
        a = self.raw_coords[0]
        b = self.raw_coords[1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]

        def cross_product(p):
            return dx * (p[1] - a[1]) - dy * (p[0] - a[0])

        cp_prev = cross_product(prev_pt)
        cp_curr = cross_product(curr_pt)

        # Tolerance epsilon for boundary
        eps = 1e-6
        if cp_prev > eps and cp_curr < -eps:
            return "CROSS_IN"
        elif cp_prev < -eps and cp_curr > eps:
            return "CROSS_OUT"

        return None


class TrackZoneState:
    """Tracks state of a specific person track_id in a specific zone."""
    def __init__(self, track_id: int, zone_id: str):
        self.track_id = track_id
        self.zone_id = zone_id
        self.consecutive_inside: int = 0
        self.is_confirmed_inside: bool = False
        self.first_inside_time: Optional[float] = None
        self.last_alert_time: Optional[float] = None
        self.loiter_alert_fired: bool = False


class ZoneManager:
    """Manages all active zones and state machines for a video stream."""
    def __init__(self):
        self.zones: Dict[str, ZoneGeometry] = {}
        # (track_id, zone_id) -> TrackZoneState
        self.track_states: Dict[Tuple[int, str], TrackZoneState] = {}
        # track_id -> previous normalized foot point (x, y)
        self.previous_positions: Dict[int, Tuple[float, float]] = {}
        # track_id -> last seen timestamp
        self.track_last_seen: Dict[int, float] = {}

    def update_zones(self, zone_models: List[Any]):
        """Replace active zones with new configuration."""
        new_zones = {}
        for z in zone_models:
            coords = z.coordinates
            if isinstance(coords, str):
                try:
                    coords = json.loads(coords)
                except Exception:
                    coords = []
            
            zg = ZoneGeometry(
                zone_id=str(z.id),
                name=z.name,
                zone_type=z.zone_type,
                coordinates=coords,
                armed=z.armed,
                schedule_start=z.schedule_start,
                schedule_end=z.schedule_end,
                color=getattr(z, "color", "#EF4444")
            )
            new_zones[zg.zone_id] = zg
        self.zones = new_zones

    def process_tracks(
        self,
        tracks: List[Dict[str, Any]],
        current_time: float,
        debounce_frames: int = 2,
        cooldown_sec: float = 8.0,
        loiter_sec: float = 6.0
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all current tracks against all active zones.
        tracks: list of dicts: {'track_id': int, 'bbox': [x1, y1, x2, y2], 'norm_foot': (nx, ny), 'confidence': float}
        Returns list of event dicts to be emitted.
        """
        events = []
        current_track_ids = set()

        for t in tracks:
            track_id = t["track_id"]
            current_track_ids.add(track_id)
            norm_foot = t["norm_foot"]
            confidence = t.get("confidence", 0.0)
            prev_pos = self.previous_positions.get(track_id)

            for zone_id, zone in self.zones.items():
                if not zone.is_armed_at():
                    continue

                state_key = (track_id, zone_id)
                if state_key not in self.track_states:
                    self.track_states[state_key] = TrackZoneState(track_id, zone_id)
                state = self.track_states[state_key]

                if zone.zone_type == "tripwire":
                    # Tripwire line check
                    if prev_pos is not None:
                        cross_dir = zone.check_tripwire_crossing(prev_pos, norm_foot)
                        if cross_dir:
                            can_alert = (
                                state.last_alert_time is None
                                or (current_time - state.last_alert_time) >= cooldown_sec
                            )
                            if can_alert:
                                state.last_alert_time = current_time
                                events.append({
                                    "event_type": cross_dir,
                                    "zone_id": zone_id,
                                    "zone_name": zone.name,
                                    "track_id": track_id,
                                    "confidence": confidence,
                                    "foot_point": norm_foot,
                                    "timestamp": current_time,
                                    "details": f"Directional tripwire crossing: {cross_dir}"
                                })

                else:
                    # Polygon zone check
                    is_inside = zone.contains_point(norm_foot[0], norm_foot[1])

                    if is_inside:
                        state.consecutive_inside += 1
                        if state.consecutive_inside >= debounce_frames:
                            if not state.is_confirmed_inside:
                                # Transition: outside -> inside => ENTER!
                                state.is_confirmed_inside = True
                                state.first_inside_time = current_time
                                state.loiter_alert_fired = False

                                # Check cooldown
                                can_alert = (
                                    state.last_alert_time is None
                                    or (current_time - state.last_alert_time) >= cooldown_sec
                                )
                                if can_alert:
                                    state.last_alert_time = current_time
                                    events.append({
                                        "event_type": "ENTER",
                                        "zone_id": zone_id,
                                        "zone_name": zone.name,
                                        "track_id": track_id,
                                        "confidence": confidence,
                                        "foot_point": norm_foot,
                                        "timestamp": current_time,
                                        "details": f"Person entered zone '{zone.name}'"
                                    })
                            else:
                                # Already inside - check loitering
                                if (
                                    not state.loiter_alert_fired
                                    and state.first_inside_time is not None
                                    and (current_time - state.first_inside_time) >= loiter_sec
                                ):
                                    state.loiter_alert_fired = True
                                    events.append({
                                        "event_type": "LOITER",
                                        "zone_id": zone_id,
                                        "zone_name": zone.name,
                                        "track_id": track_id,
                                        "confidence": confidence,
                                        "foot_point": norm_foot,
                                        "timestamp": current_time,
                                        "details": f"Person loitering in '{zone.name}' for {loiter_sec}s"
                                    })
                    else:
                        # Outside zone
                        if state.is_confirmed_inside:
                            # Transition: inside -> outside => EXIT!
                            state.is_confirmed_inside = False
                            state.consecutive_inside = 0
                            state.first_inside_time = None
                            state.loiter_alert_fired = False
                            events.append({
                                "event_type": "EXIT",
                                "zone_id": zone_id,
                                "zone_name": zone.name,
                                "track_id": track_id,
                                "confidence": confidence,
                                "foot_point": norm_foot,
                                "timestamp": current_time,
                                "details": f"Person exited zone '{zone.name}'"
                            })
                        else:
                            state.consecutive_inside = 0

            # Store position and timestamp
            self.previous_positions[track_id] = norm_foot
            self.track_last_seen[track_id] = current_time

        # Cleanup lost tracks (not seen for > 10 seconds)
        lost_tracks = [
            tid for tid, last_t in self.track_last_seen.items()
            if (current_time - last_t) > 10.0
        ]
        for tid in lost_tracks:
            self.track_last_seen.pop(tid, None)
            self.previous_positions.pop(tid, None)
            keys_to_del = [k for k in self.track_states.keys() if k[0] == tid]
            for k in keys_to_del:
                # If was confirmed inside, emit EXIT on disappearance
                st = self.track_states[k]
                if st.is_confirmed_inside and k[1] in self.zones:
                    events.append({
                        "event_type": "EXIT",
                        "zone_id": k[1],
                        "zone_name": self.zones[k[1]].name,
                        "track_id": tid,
                        "confidence": 0.0,
                        "foot_point": (0.0, 0.0),
                        "timestamp": current_time,
                        "details": f"Person {tid} left zone (track lost)"
                    })
                del self.track_states[k]

        return events
