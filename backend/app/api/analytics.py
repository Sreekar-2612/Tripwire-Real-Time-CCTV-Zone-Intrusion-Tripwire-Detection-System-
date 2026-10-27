from datetime import datetime, timedelta
from typing import Dict, Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.db import get_db
from backend.app.models import Event, Zone, Stream

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@router.get("/summary")
def get_analytics_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    now = datetime.utcnow()
    past_24h = now - timedelta(hours=24)
    past_7d = now - timedelta(days=7)

    total_all_time = db.query(func.count(Event.id)).scalar() or 0
    total_24h = db.query(func.count(Event.id)).filter(Event.timestamp >= past_24h).scalar() or 0
    total_7d = db.query(func.count(Event.id)).filter(Event.timestamp >= past_7d).scalar() or 0

    # Breakdown by event type (ENTER, LOITER, CROSS_IN, EXIT)
    type_counts = (
        db.query(Event.event_type, func.count(Event.id))
        .filter(Event.timestamp >= past_7d)
        .group_by(Event.event_type)
        .all()
    )
    event_type_distribution = {t: c for t, c in type_counts}

    # Breakdown by zone
    zone_counts = (
        db.query(Zone.name, func.count(Event.id))
        .join(Event, Zone.id == Event.zone_id)
        .filter(Event.timestamp >= past_7d)
        .group_by(Zone.name)
        .order_by(func.count(Event.id).desc())
        .limit(6)
        .all()
    )
    zone_distribution = [{"zone_name": z_name, "count": c} for z_name, c in zone_counts]

    # Hourly distribution past 24h (0..23)
    hourly_events = [0] * 24
    recent_events = db.query(Event.timestamp).filter(Event.timestamp >= past_24h).all()
    for ev in recent_events:
        h = ev.timestamp.hour
        hourly_events[h] += 1

    active_streams_count = db.query(func.count(Stream.id)).filter(Stream.is_active == True).scalar() or 0
    armed_zones_count = db.query(func.count(Zone.id)).filter(Zone.armed == True).scalar() or 0

    return {
        "total_all_time": total_all_time,
        "total_24h": total_24h,
        "total_7d": total_7d,
        "active_streams_count": active_streams_count,
        "armed_zones_count": armed_zones_count,
        "event_type_distribution": event_type_distribution,
        "zone_distribution": zone_distribution,
        "hourly_distribution": hourly_events,
        "system_status": "SECURE" if total_24h == 0 else "ACTIVITY_RECORDED"
    }
