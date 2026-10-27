import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from backend.app.config import settings
from backend.app.db import init_db, SessionLocal
from backend.app.models import Stream, Zone
from backend.app.core.pipeline import StreamPipeline

def run_demo():
    print("=" * 60)
    print("TRIPWIRE PHASE 1 DEMO: File-Source Pipeline & Event Triggering")
    print("=" * 60)
    init_db()

    video_path = str(settings.RECORDINGS_DIR / "cctv_sample.mp4")
    if not os.path.exists(video_path):
        print(f"Error: Sample video {video_path} not found.")
        return

    db = SessionLocal()
    # Check or create demo stream
    stream = db.query(Stream).filter(Stream.name == "Demo Surveillance Camera").first()
    if not stream:
        stream = Stream(
            name="Demo Surveillance Camera",
            source_type="file",
            source_url=video_path,
            is_active=True,
            loop_file=False
        )
        db.add(stream)
        db.commit()
        db.refresh(stream)

    # Check or create demo zone in center where person walks
    # Normalized coords: center region x from 0.35 to 0.65, y from 0.4 to 0.95
    zone = db.query(Zone).filter(Zone.stream_id == stream.id).first()
    if not zone:
        zone = Zone(
            stream_id=stream.id,
            name="Restricted Courtyard Core",
            zone_type="polygon",
            coordinates="[[0.35, 0.4], [0.65, 0.4], [0.65, 0.95], [0.35, 0.95]]",
            armed=True,
            color="#EF4444"
        )
        db.add(zone)
        db.commit()
    stream_id = stream.id
    stream_name = stream.name
    zone_name = zone.name
    db.close()

    print(f"Loaded Stream '{stream_name}' with Zone '{zone_name}'")
    print("Starting stream pipeline processing...")

    captured_events = []
    def on_ws_alert(data):
        captured_events.append(data)
        print(f"[LIVE WS ALERT RECEIVED] Type: {data.get('type')} | Event: {data.get('event_type')} | Msg: {data.get('message')}")

    pipeline = StreamPipeline(
        stream_id=stream_id,
        name=stream_name,
        source_type="file",
        source_url=video_path,
        loop_file=False,
        ws_broadcast_callback=on_ws_alert
    )
    pipeline.start()

    # Wait for video playback (sample video is 10s)
    start_t = time.time()
    while pipeline.is_running and (time.time() - start_t) < 14:
        time.sleep(0.5)

    pipeline.stop()
    print("=" * 60)
    print(f"Pipeline finished. Total live alerts broadcast: {len(captured_events)}")
    for i, e in enumerate(captured_events, 1):
        print(f"  Alert #{i}: {e['event_type']} - {e['message']}")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()
