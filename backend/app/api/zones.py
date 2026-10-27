import json
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.models import Stream, Zone
from backend.app.schemas import ZoneBase, ZoneResponse
from backend.app.api.streams import active_pipelines

logger = logging.getLogger("tripwire.api.zones")
router = APIRouter(prefix="/api/streams/{stream_id}/zones", tags=["zones"])

@router.get("", response_model=List[ZoneResponse])
def get_zones(stream_id: str, db: Session = Depends(get_db)):
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    zones = db.query(Zone).filter(Zone.stream_id == stream_id).all()
    results = []
    for z in zones:
        coords = z.coordinates
        if isinstance(coords, str):
            try:
                coords = json.loads(coords)
            except Exception:
                coords = []
        results.append(
            ZoneResponse(
                id=z.id,
                stream_id=z.stream_id,
                name=z.name,
                zone_type=z.zone_type,
                coordinates=coords,
                armed=z.armed,
                color=z.color,
                schedule_start=z.schedule_start,
                schedule_end=z.schedule_end,
                created_at=z.created_at
            )
        )
    return results

@router.put("", response_model=List[ZoneResponse])
def update_zones(stream_id: str, zones_in: List[ZoneBase], db: Session = Depends(get_db)):
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    # Replace existing zones for this stream
    db.query(Zone).filter(Zone.stream_id == stream_id).delete()

    created_zones = []
    for zb in zones_in:
        new_zone = Zone(
            stream_id=stream_id,
            name=zb.name,
            zone_type=zb.zone_type,
            coordinates=json.dumps(zb.coordinates),
            armed=zb.armed,
            color=zb.color,
            schedule_start=zb.schedule_start,
            schedule_end=zb.schedule_end
        )
        db.add(new_zone)
        created_zones.append(new_zone)

    db.commit()

    # Inform running pipeline to hot-reload its zones
    if stream_id in active_pipelines:
        active_pipelines[stream_id].reload_zones()

    # Return refreshed zones
    results = []
    for z in created_zones:
        db.refresh(z)
        results.append(
            ZoneResponse(
                id=z.id,
                stream_id=z.stream_id,
                name=z.name,
                zone_type=z.zone_type,
                coordinates=json.loads(z.coordinates),
                armed=z.armed,
                color=z.color,
                schedule_start=z.schedule_start,
                schedule_end=z.schedule_end,
                created_at=z.created_at
            )
        )
    return results

@router.delete("/{zone_id}")
def delete_single_zone(stream_id: str, zone_id: str, db: Session = Depends(get_db)):
    zone = db.query(Zone).filter(Zone.stream_id == stream_id, Zone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")

    db.delete(zone)
    db.commit()

    if stream_id in active_pipelines:
        active_pipelines[stream_id].reload_zones()

    return {"message": "Zone deleted"}
