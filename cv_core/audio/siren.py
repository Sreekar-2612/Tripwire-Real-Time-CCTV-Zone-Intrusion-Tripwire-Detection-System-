import sys
import time
import threading
from typing import Dict, List, Optional, Set, Tuple

# Try importing winsound for Windows desktop sound support
try:
    import winsound
except ImportError:
    winsound = None


class SirenAlertController:
    """
    Central Threat Audio & Audible Siren Controller for TRIP-WIRE.
    
    Provides:
    - Audible alarm configuration toggle (enabled vs muted)
    - Direct state synchronization with ZoneMonitor events:
        * Breaches: CROSS_IN, CROSS_OUT, LOITER
        * High-threat kinetics: SPEED_START, ACTION_START
    - Clean Web Audio API synthesis for web HUD (Streamlit / FastAPI)
    - Asynchronous non-blocking desktop beeps via daemon threads to prevent
      video capture frame rate / FPS degradation.
    """

    BREACH_EVENTS: Set[str] = {"CROSS_IN", "CROSS_OUT", "LOITER"}
    KINETIC_THREAT_EVENTS: Set[str] = {"SPEED_START", "ACTION_START"}
    ALARM_TRIGGER_EVENTS: Set[str] = BREACH_EVENTS | KINETIC_THREAT_EVENTS

    def __init__(self, enabled: bool = True, sound_cooldown_sec: float = 0.85):
        self.enabled: bool = enabled
        self.sound_cooldown_sec: float = sound_cooldown_sec
        self.last_sound_time: float = 0.0
        self.is_alarm_active: bool = False
        self.active_breach_count: int = 0
        self.active_reasons: List[str] = []
        self._sound_thread: Optional[threading.Thread] = None

    def set_enabled(self, enabled: bool) -> None:
        """Sets the audible alarm state (enabled vs muted)."""
        self.enabled = bool(enabled)

    def toggle_mute(self) -> bool:
        """Toggles the mute state. Returns the new enabled state."""
        self.enabled = not self.enabled
        return self.enabled

    @property
    def is_muted(self) -> bool:
        """Returns True if the audible alarm is muted."""
        return not self.enabled

    def sync_with_events(
        self,
        events: Optional[List[Dict]] = None,
        states: Optional[Dict] = None,
        breach_count: Optional[int] = None
    ) -> bool:
        """
        Synchronizes alarm state with ZoneMonitor outputs and track states.
        
        Args:
            events: List of event dicts emitted by ZoneMonitor.update(), e.g.:
                    {'track_id': 1, 'event': 'CROSS_IN', 'zone_id': 'tw_1', 'reason': '...'}
            states: Track states dict from ZoneMonitor.update(), mapping track_id to:
                    {'is_breaching': bool, 'is_monitoring': bool, 'reasons': [...]}
            breach_count: Explicit count of active breaching targets in frame.
            
        Returns:
            bool: True if an alarm state is currently active, False otherwise.
        """
        reasons: List[str] = []
        active = False

        if breach_count is not None:
            self.active_breach_count = max(0, int(breach_count))
            if self.active_breach_count > 0:
                active = True

        if states:
            active_breaches = 0
            for tid, s in states.items():
                if s.get("is_breaching"):
                    active = True
                    active_breaches += 1
                    track_reasons = s.get("reasons", [])
                    reasons.extend(track_reasons)
            if breach_count is None:
                self.active_breach_count = active_breaches

        if events:
            for ev in events:
                event_type = ev.get("event")
                if event_type in self.ALARM_TRIGGER_EVENTS:
                    active = True
                    reason = ev.get("reason", f"Event: {event_type}")
                    if reason not in reasons:
                        reasons.append(reason)

        self.is_alarm_active = active
        self.active_reasons = reasons
        return self.is_alarm_active

    def generate_web_audio_component(self, force: bool = False, now: Optional[float] = None) -> str:
        """
        Generates the HTML / Web Audio API snippet synthesizing an alternating
        880Hz / 659Hz security alert siren tone.
        
        Returns an empty string if:
        - The audible siren is muted (enabled == False)
        - No breach is active (and force is False)
        - Sound cooldown period has not elapsed (and force is False)
        """
        if not self.enabled:
            return ""

        if not self.is_alarm_active and not force:
            return ""

        ts = now if now is not None else time.time()
        if not force and (ts - self.last_sound_time) < self.sound_cooldown_sec:
            return ""

        self.last_sound_time = ts
        alarm_token = int(ts * 1000)

        return f"""
<div id="tripwire-siren-{alarm_token}" style="display:none;"></div>
<script>
(function() {{
    try {{
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return;
        const ctx = new AudioContextClass();
        if (ctx.state === 'suspended') {{
            ctx.resume();
        }}
        const t = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        // Crisp two-tone tactical security siren: alternating 880Hz / 659Hz
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, t);
        osc.frequency.setValueAtTime(659, t + 0.35);

        gain.gain.setValueAtTime(0.0001, t);
        gain.gain.exponentialRampToValueAtTime(0.28, t + 0.05);
        gain.gain.setValueAtTime(0.28, t + 0.65);
        gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.80);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(t);
        osc.stop(t + 0.82);
        setTimeout(() => {{
            try {{ ctx.close(); }} catch(e) {{}}
        }}, 1000);
    }} catch (err) {{
        console.warn('Tripwire Web Audio API notice:', err);
    }}
}})();
</script>
"""

    def trigger_non_blocking_beep(self, now: Optional[float] = None) -> bool:
        """
        Triggers a non-blocking tactical beep in a background daemon thread.
        Guarantees zero blocking of the main video frame processing loop,
        maintaining 100% video FPS throughput.
        
        Returns:
            bool: True if tone was scheduled, False if muted, inactive, or on cooldown.
        """
        if not self.enabled or not self.is_alarm_active:
            return False

        ts = now if now is not None else time.time()
        if (ts - self.last_sound_time) < self.sound_cooldown_sec:
            return False

        self.last_sound_time = ts

        def _worker():
            try:
                if winsound is not None:
                    # Alternating two-tone beep: 880Hz high, 659Hz low
                    winsound.Beep(880, 120)
                    winsound.Beep(659, 120)
                else:
                    sys.stdout.write("\a")
                    sys.stdout.flush()
            except Exception:
                pass

        self._sound_thread = threading.Thread(target=_worker, daemon=True, name="TripwireAudioSiren")
        self._sound_thread.start()
        return True
