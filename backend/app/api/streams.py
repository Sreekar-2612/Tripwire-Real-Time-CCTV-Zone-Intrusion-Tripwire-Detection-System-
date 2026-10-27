import os
import time
import asyncio
import logging
from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db import get_db, SessionLocal
from backend.app.models import Stream, Zone
from backend.app.schemas import StreamCreate, StreamResponse
from backend.app.core.pipeline import StreamPipeline
from backend.app.api.ws import ws_manager

logger = logging.getLogger("tripwire.api.streams")
router = APIRouter(prefix="/api/streams", tags=["streams"])

# Global map of active stream pipelines: stream_id -> StreamPipeline
active_pipelines: Dict[str, StreamPipeline] = {}

def broadcast_ws_sync(data: dict):
    """Bridge synchronous pipeline threads with asynchronous WebSocket manager."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast_json(data), loop)
        else:
            loop.run_until_complete(ws_manager.broadcast_json(data))
    except Exception:
        pass

def get_or_create_pipeline(stream: Stream) -> StreamPipeline:
    if stream.id not in active_pipelines:
        pipeline = StreamPipeline(
            stream_id=stream.id,
            name=stream.name,
            source_type=stream.source_type,
            source_url=stream.source_url,
            loop_file=stream.loop_file,
            ws_broadcast_callback=broadcast_ws_sync
        )
        active_pipelines[stream.id] = pipeline
        if stream.is_active:
            pipeline.start()
    return active_pipelines[stream.id]

def init_active_streams():
    """Initializes and boots all active streams configured in database on startup."""
    db = SessionLocal()
    try:
        streams = db.query(Stream).filter(Stream.is_active == True).all()
        for s in streams:
            try:
                get_or_create_pipeline(s)
            except Exception as e:
                logger.error(f"Failed to auto-start stream {s.name}: {e}")
    finally:
        db.close()


@router.get("", response_model=List[StreamResponse])
def list_streams(db: Session = Depends(get_db)):
    streams = db.query(Stream).all()
    # Populate live FPS if running
    res = []
    for s in streams:
        sr = StreamResponse.model_validate(s)
        if s.id in active_pipelines and active_pipelines[s.id].is_running:
            sr.fps = round(active_pipelines[s.id].fps_meter, 1)
        res.append(sr)
    return res


@router.post("", response_model=StreamResponse)
def create_stream(stream_in: StreamCreate, db: Session = Depends(get_db)):
    new_stream = Stream(
        name=stream_in.name,
        source_type=stream_in.source_type,
        source_url=stream_in.source_url,
        is_active=stream_in.is_active,
        loop_file=stream_in.loop_file
    )
    db.add(new_stream)
    db.commit()
    db.refresh(new_stream)

    # Start pipeline
    pipeline = get_or_create_pipeline(new_stream)
    if new_stream.is_active:
        pipeline.start()

    return StreamResponse.model_validate(new_stream)


@router.post("/upload", response_model=StreamResponse)
async def upload_video_stream(
    name: str = Form(...),
    loop_file: bool = Form(True),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Save uploaded file
    filename = f"{int(time.time())}_{file.filename}"
    filepath = settings.UPLOADS_DIR / filename
    with open(filepath, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    new_stream = Stream(
        name=name,
        source_type="file",
        source_url=str(filepath),
        is_active=True,
        loop_file=loop_file
    )
    db.add(new_stream)
    db.commit()
    db.refresh(new_stream)

    pipeline = get_or_create_pipeline(new_stream)
    pipeline.start()

    return StreamResponse.model_validate(new_stream)


@router.get("/{stream_id}", response_model=StreamResponse)
def get_stream(stream_id: str, db: Session = Depends(get_db)):
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    sr = StreamResponse.model_validate(stream)
    if stream.id in active_pipelines:
        sr.fps = round(active_pipelines[stream.id].fps_meter, 1)
    return sr


@router.delete("/{stream_id}")
def delete_stream(stream_id: str, db: Session = Depends(get_db)):
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    # Stop and delete pipeline
    if stream_id in active_pipelines:
        active_pipelines[stream_id].stop()
        del active_pipelines[stream_id]

    db.delete(stream)
    db.commit()
    return {"message": "Stream deleted successfully"}


@router.post("/{stream_id}/start")
def start_stream(stream_id: str, db: Session = Depends(get_db)):
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    stream.is_active = True
    db.commit()

    pipeline = get_or_create_pipeline(stream)
    pipeline.start()
    return {"status": "started", "stream_id": stream_id}


@router.post("/{stream_id}/stop")
def stop_stream(stream_id: str, db: Session = Depends(get_db)):
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    stream.is_active = False
    db.commit()

    if stream_id in active_pipelines:
        active_pipelines[stream_id].stop()
    return {"status": "stopped", "stream_id": stream_id}


@router.get("/{stream_id}/mjpeg")
def stream_mjpeg(stream_id: str, db: Session = Depends(get_db)):
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    pipeline = get_or_create_pipeline(stream)
    if not pipeline.is_running:
        pipeline.start()

    def mjpeg_generator():
        # Keep yielding latest JPEG frame with boundary
        while True:
            jpeg = pipeline.get_latest_jpeg()
            if jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.04)  # ~25 FPS delivery

    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
