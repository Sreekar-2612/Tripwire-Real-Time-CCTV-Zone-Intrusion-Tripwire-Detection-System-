import os
import csv
import io
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.models import Event, Zone, Stream
from backend.app.schemas import EventResponse

logger = logging.getLogger("tripwire.api.events")
router = APIRouter(prefix="/api/events", tags=["events"])

@router.get("", response_model=List[EventResponse])
def list_events(
    stream_id: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    from_date: Optional[datetime] = Query(None, alias="from"),
    to_date: Optional[datetime] = Query(None, alias="to"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    query = db.query(Event, Zone.name.label("zone_name")).outerjoin(Zone, Event.zone_id == Zone.id)

    if stream_id:
        query = query.filter(Event.stream_id == stream_id)
    if event_type:
        query = query.filter(Event.event_type == event_type)
    if from_date:
        query = query.filter(Event.timestamp >= from_date)
    if to_date:
        query = query.filter(Event.timestamp <= to_date)

    query = query.order_by(Event.timestamp.desc()).offset(offset).limit(limit)
    rows = query.all()

    results = []
    for ev, z_name in rows:
        results.append(
            EventResponse(
                id=ev.id,
                stream_id=ev.stream_id,
                zone_id=ev.zone_id,
                zone_name=z_name or "Zone",
                track_id=ev.track_id,
                event_type=ev.event_type,
                timestamp=ev.timestamp,
                confidence=round(ev.confidence, 2),
                snapshot_url=f"/api/events/{ev.id}/snapshot" if ev.snapshot_path and os.path.exists(ev.snapshot_path) else None,
                clip_url=f"/api/events/{ev.id}/clip" if ev.clip_path and os.path.exists(ev.clip_path) else None,
                details=ev.details
            )
        )
    return results

@router.get("/export/csv")
def export_events_csv(
    stream_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(Event, Zone.name.label("zone_name")).outerjoin(Zone, Event.zone_id == Zone.id)
    if stream_id:
        query = query.filter(Event.stream_id == stream_id)
    query = query.order_by(Event.timestamp.desc())

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Event ID", "Timestamp (UTC)", "Stream ID", "Zone Name", "Track ID", "Event Type", "Confidence"])

    for ev, z_name in query.all():
        writer.writerow([
            ev.id,
            ev.timestamp.isoformat(),
            ev.stream_id,
            z_name or "N/A",
            ev.track_id,
            ev.event_type,
            round(ev.confidence, 2)
        ])

    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=tripwire_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )

@router.get("/{event_id}", response_model=EventResponse)
def get_event(event_id: str, db: Session = Depends(get_db)):
    res = db.query(Event, Zone.name.label("zone_name")).outerjoin(Zone, Event.zone_id == Zone.id).filter(Event.id == event_id).first()
    if not res:
        raise HTTPException(status_code=404, detail="Event not found")
    
    ev, z_name = res
    return EventResponse(
        id=ev.id,
        stream_id=ev.stream_id,
        zone_id=ev.zone_id,
        zone_name=z_name or "Zone",
        track_id=ev.track_id,
        event_type=ev.event_type,
        timestamp=ev.timestamp,
        confidence=round(ev.confidence, 2),
        snapshot_url=f"/api/events/{ev.id}/snapshot" if ev.snapshot_path and os.path.exists(ev.snapshot_path) else None,
        clip_url=f"/api/events/{ev.id}/clip" if ev.clip_path and os.path.exists(ev.clip_path) else None,
        details=ev.details
    )

@router.get("/{event_id}/snapshot")
def get_event_snapshot(event_id: str, db: Session = Depends(get_db)):
    ev = db.query(Event).filter(Event.id == event_id).first()
    if not ev or not ev.snapshot_path or not os.path.exists(ev.snapshot_path):
        raise HTTPException(status_code=404, detail="Snapshot image not found")
    return FileResponse(ev.snapshot_path, media_type="image/jpeg")

@router.get("/{event_id}/clip")
def get_event_clip(event_id: str, db: Session = Depends(get_db)):
    ev = db.query(Event).filter(Event.id == event_id).first()
    if not ev or not ev.clip_path or not os.path.exists(ev.clip_path):
        raise HTTPException(status_code=404, detail="Clip video not ready or not found")
    return FileResponse(ev.clip_path, media_type="video/mp4")

@router.delete("/{event_id}")
def delete_event(event_id: str, db: Session = Depends(get_db)):
    ev = db.query(Event).filter(Event.id == event_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")

    # Clean up disk files
    if ev.snapshot_path and os.path.exists(ev.snapshot_path):
        try:
            os.remove(ev.snapshot_path)
        except Exception:
            pass
    if ev.clip_path and os.path.exists(ev.clip_path):
        try:
            os.remove(ev.clip_path)
        except Exception:
            pass

    db.delete(ev)
    db.commit()
    return {"message": "Event deleted"}
