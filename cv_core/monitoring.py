import numpy as np


class ZoneMonitor:
    """Per-camera zone history and transition events, evaluated once per frame."""

    def __init__(self):
        self.time = 0.0
        self.inside = {}
        self.wire_points = {}
        self.latches = {}
        self.loitering = set()
        self.actions = set()

    def update(self, tracks, zones, dt, dwell_confirm_sec=0.0, speed_alarm=float('inf')):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Frame interval must be finite and positive')
        self.time += dt
        states, events = {}, []
        inside_now, wires_now, actions_now = set(), set(), set()

        def emit(tid, event, zone, reason):
            events.append(dict(track_id=tid, event=event, zone_id=zone, reason=reason))

        for track in tracks:
            tid = track['track_id']
            reasons, monitoring = [], []
            pos = track.get('world_pos')
            if pos is not None and np.all(np.isfinite(pos)):
                for zone in zones:
                    if not zone.armed:
                        continue
                    key = (tid, zone.zone_id)
                    if zone.zone_type == 'tripwire':
                        wires_now.add(key)
                        # Retain the last off-wire point: touching and returning is not a crossing.
                        a, b = zone.world_coords[:2]
                        side = (b[0]-a[0])*(pos[1]-a[1]) - (b[1]-a[1])*(pos[0]-a[0])
                        previous = self.wire_points.get(key)
                        if abs(side) > 1e-5:
                            crossing = zone.check_tripwire_crossing(previous, pos) if previous is not None else None
                            self.wire_points[key] = pos
                            if crossing:
                                self.latches[key] = self.time
                                emit(tid, crossing, zone.zone_id, f'Crossed {zone.name}')
                        if self.time - self.latches.get(key, float('-inf')) < 3:
                            reasons.append(f'Crossed {zone.name}')
                    elif zone.contains_point(*pos):
                        inside_now.add(key)
                        if key not in self.inside:
                            self.inside[key] = (self.time, zone.name)
                            emit(tid, 'ENTER', zone.zone_id, f'Entered {zone.name}')
                        dwell = self.time - self.inside[key][0]
                        if dwell >= dwell_confirm_sec:
                            reasons.append(f'Loitering in {zone.name}')
                            if key not in self.loitering:
                                self.loitering.add(key)
                                emit(tid, 'LOITER', zone.zone_id, f'Loitering in {zone.name}')
                        else:
                            monitoring.append(f'Inside {zone.name} ({dwell:.1f}s / {dwell_confirm_sec:.1f}s)')

            if pos is not None and track.get('speed', 0) > speed_alarm:
                key = (tid, 'speed')
                actions_now.add(key)
                reasons.append('Speed threshold exceeded')
                if key not in self.actions:
                    emit(tid, 'SPEED_START', 'speed', reasons[-1])
            if track.get('threat_level', 1) >= 8:
                posture = track.get('posture', 'unknown')
                key = (tid, posture)
                actions_now.add(key)
                reasons.append(f'High threat action: {posture.upper()}')
                if key not in self.actions:
                    emit(tid, 'ACTION_START', posture, reasons[-1])
            states[tid] = dict(is_breaching=bool(reasons), is_monitoring=bool(monitoring),
                               reasons=reasons or monitoring)

        for key in self.inside.keys() - inside_now:
            _, name = self.inside.pop(key)
            self.loitering.discard(key)
            emit(key[0], 'EXIT', key[1], f'Left {name} or tracking lost')
        self.wire_points = {k: v for k, v in self.wire_points.items() if k in wires_now}
        self.latches = {k: v for k, v in self.latches.items() if k in wires_now and self.time-v < 3}
        for tid, action in self.actions - actions_now:
            emit(tid, 'ACTION_END' if action != 'speed' else 'SPEED_END', action, f'{action} ended')
        self.actions = actions_now
        return states, events
