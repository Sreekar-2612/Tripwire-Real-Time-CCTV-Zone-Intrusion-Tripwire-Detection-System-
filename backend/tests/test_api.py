import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "active_streams" in data
    assert "system" in data

def test_prometheus_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "tripwire_active_streams" in text
    assert "tripwire_armed_zones" in text
    assert "tripwire_system_cpu_percent" in text

def test_streams_api():
    response = client.get("/api/streams")
    assert response.status_code == 200
    streams = response.json()
    assert isinstance(streams, list)

def test_settings_api():
    # Read settings
    res = client.get("/api/settings")
    assert res.status_code == 200
    cfg = res.json()
    assert "confidence_threshold" in cfg
    assert "cooldown_seconds" in cfg

    # Update settings
    cfg["confidence_threshold"] = 0.45
    put_res = client.put("/api/settings", json=cfg)
    assert put_res.status_code == 200
    updated_cfg = put_res.json()
    assert updated_cfg["confidence_threshold"] == 0.45

def test_events_api():
    response = client.get("/api/events?limit=10")
    assert response.status_code == 200
    events = response.json()
    assert isinstance(events, list)

def test_analytics_summary_api():
    response = client.get("/api/analytics/summary")
    assert response.status_code == 200
    summary = response.json()
    assert "total_all_time" in summary
    assert "hourly_distribution" in summary
    assert len(summary["hourly_distribution"]) == 24
    assert "event_type_distribution" in summary
