import time
from unittest.mock import Mock, patch
from io import BytesIO
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from scripts import web_demo as web
from scripts.demo_gui import TripwireInteractiveDemo
from cv_core.monitoring import ZoneMonitor
from cv_core.geometry.zones_3d import MetricZone3D


def test_web_alarm_status_endpoint_and_html_siren():
    """Tests the /api/alarm_status endpoint and presence of Web Audio synthesis in HTML."""
    client = TestClient(web.app)
    
    # Check default secure state
    web.engine.active_breach = False
    web.engine.active_breach_count = 0
    resp = client.get("/api/alarm_status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_breach"] is False
    assert data["breach_count"] == 0

    # Simulate active breach in web engine
    web.engine.active_breach = True
    web.engine.active_breach_count = 2
    resp2 = client.get("/api/alarm_status")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["active_breach"] is True
    assert data2["breach_count"] == 2

    # Check dashboard HTML contains Web Audio synthesizer and alert banner
    page = client.get("/").text
    assert "alertBanner" in page
    assert "CRITICAL PERIMETER BREACH ACTIVE" in page
    assert "audioAlarmBtn" in page
    assert "AudioContext" in page
    assert "880" in page and "659" in page
    assert "playTacticalSiren" in page

    # Reset
    web.engine.active_breach = False
    web.engine.active_breach_count = 0


def test_demo_gui_audio_alarm_threading_and_mute():
    """Tests demo_gui audio alarm non-blocking thread dispatch and mute behavior."""
    demo = TripwireInteractiveDemo.__new__(TripwireInteractiveDemo)
    demo.audio_alarm_enabled = True
    demo.last_alarm_sound_time = 0.0

    mock_beep = Mock()
    with patch("scripts.demo_gui.winsound") as mock_ws:
        mock_ws.Beep = mock_beep

        # Trigger alarm
        t_start = time.time()
        demo._trigger_audio_alarm()
        duration = time.time() - t_start
        # Must be non-blocking (well under 50ms)
        assert duration < 0.05
        # Give daemon thread a brief moment to run
        time.sleep(0.05)
        assert mock_beep.call_count >= 1

        # Second trigger within throttle interval (< 0.65s) should be suppressed
        call_count_before = mock_beep.call_count
        demo._trigger_audio_alarm()
        time.sleep(0.05)
        assert mock_beep.call_count == call_count_before

        # Reset throttle and test mute toggle
        demo.last_alarm_sound_time = 0.0
        demo.audio_alarm_enabled = False
        demo._trigger_audio_alarm()
        time.sleep(0.05)
        assert mock_beep.call_count == call_count_before


def test_streamlit_audio_siren_and_visual_banner():
    """Tests that app_streamlit contains audio siren toggle, error banner, and Web Audio synthesizer."""
    app_code = Path("app_streamlit.py").read_text(encoding="utf-8")
    
    # 1. Sidebar control
    assert "🔊 Audible Alarm Siren" in app_code
    assert "enable_audio_alarm" in app_code
    
    # 2. Visual pulsing error banner
    assert "CRITICAL PERIMETER BREACH ACTIVE — INTRUDER DETECTED IN RESTRICTED ZONE" in app_code
    assert "alert_banner_placeholder.error" in app_code
    
    # 3. Web Audio API synthesizer component
    assert "audio_siren_placeholder" in app_code
    assert "AudioContextClass" in app_code or "AudioContext" in app_code
    assert "880" in app_code and "659" in app_code
    assert "components.html" in app_code
