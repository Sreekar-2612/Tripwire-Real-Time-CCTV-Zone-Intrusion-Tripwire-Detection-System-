import cv2
import time
import queue
import logging
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db import SessionLocal
from backend.app.models import Event, Zone, Stream
from backend.app.core.detector import PersonDetectorTracker
from backend.app.core.zones import ZoneManager
from backend.app.core.sources import BaseVideoSource, OpenCVVideoSource
from backend.app.core.recorder import FrameRingBuffer, ClipRecorder
from backend.app.core.alerts import AlertManager

logger = logging.getLogger("tripwire.pipeline")

class StreamPipeline:
    """
    Real-time vision pipeline for a single video stream:
    Source -> YOLOv8n + ByteTrack -> Zone Analysis -> Recording -> Alerting -> MJPEG.
    """
    def __init__(
        self,
        stream_id: str,
        name: str,
        source_type: str,
        source_url: str,
        loop_file: bool = True,
        ws_broadcast_callback=None
    ):
        self.stream_id = stream_id
        self.name = name
        self.source_type = source_type
        self.source_url = source_url
        self.loop_file = loop_file
        self.ws_broadcast = ws_broadcast_callback

        # Initialize CV components
        self.detector = PersonDetectorTracker(
            confidence=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            imgsz=settings.IMGSZ
        )
        self.zone_manager = ZoneManager()
        self.source: Optional[BaseVideoSource] = None

        # Buffer for clip recording & MJPEG
        self.ring_buffer = FrameRingBuffer(
            max_seconds=settings.PRE_BUFFER_SECONDS,
            fps=settings.TARGET_FPS
        )
        self.latest_jpeg_frame: Optional[bytes] = None
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # Active clip recorders subscribers
        self.clip_subscribers: List[queue.Queue] = []
        self.subscriber_lock = threading.Lock()

        # Pipeline state
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.fps_meter = 0.0
        self.people_count = 0
        self.alarm_active_until = 0.0

        # Reload zones from DB
        self.reload_zones()

    def reload_zones(self):
        """Loads latest zone polygons from SQLite database."""
        db: Session = SessionLocal()
        try:
            zones = db.query(Zone).filter(Zone.stream_id == self.stream_id).all()
            self.zone_manager.update_zones(zones)
            logger.info(f"Loaded {len(zones)} zones for stream '{self.name}' ({self.stream_id})")
        except Exception as e:
            logger.error(f"Error loading zones from DB: {e}")
        finally:
            db.close()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.source = OpenCVVideoSource(
            source_id=self.stream_id,
            source_url=self.source_url,
            source_type=self.source_type,
            loop=self.loop_file
        )
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info(f"Pipeline started for stream '{self.name}'")

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.source:
            self.source.release()
        logger.info(f"Pipeline stopped for stream '{self.name}'")

    def _run_loop(self):
        frame_idx = 0
        fps_timer = time.time()
        fps_frame_counter = 0

        while self.is_running:
            ret, frame = self.source.read_frame()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            current_time = time.time()
            fps_frame_counter += 1
            if (current_time - fps_timer) >= 1.0:
                self.fps_meter = fps_frame_counter / (current_time - fps_timer)
                fps_timer = current_time
                fps_frame_counter = 0

            # Store in ring buffer before heavy processing
            self.ring_buffer.append(frame, current_time)

            # Broadcast frame to any active clip recorder queues
            with self.subscriber_lock:
                for q in self.clip_subscribers:
                    try:
                        q.put_nowait(frame)
                    except queue.Full:
                        pass

            # Frame skipping (optional optimization)
            frame_idx += 1
            if settings.FRAME_SKIP > 0 and (frame_idx % (settings.FRAME_SKIP + 1) != 0):
                continue

            # Run detection & tracking
            tracks = self.detector.track(frame)
            self.people_count = len(tracks)

            # Process zones
            events = self.zone_manager.process_tracks(
                tracks=tracks,
                current_time=current_time,
                debounce_frames=settings.DEBOUNCE_FRAMES,
                cooldown_sec=settings.COOLDOWN_SECONDS,
                loiter_sec=settings.LOITERING_SECONDS
            )

            # Check if any alarm active
            breaching_zones = set()
            for evt in events:
                if evt["event_type"] in ["ENTER", "LOITER", "CROSS_IN", "CROSS_OUT"]:
                    self.alarm_active_until = current_time + 2.0
                    breaching_zones.add(evt.get("zone_id"))

            # Render visual annotations
            annotated_frame = self._annotate_frame(
                frame=frame.copy(),
                tracks=tracks,
                current_time=current_time,
                breaching_zones=breaching_zones
            )

            # Handle emitted events
            for evt in events:
                self._handle_event(evt, annotated_frame)

            # Update latest frame for MJPEG stream
            try:
                ret_enc, jpeg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret_enc:
                    with self.frame_lock:
                        self.latest_jpeg_frame = jpeg.tobytes()
                        self.latest_raw_frame = annotated_frame
            except Exception as e:
                logger.error(f"Error encoding MJPEG frame: {e}")

    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        current_time: float,
        breaching_zones: set
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Draw Zones
        for zid, zg in self.zone_manager.zones.items():
            is_breached = zid in breaching_zones or current_time < self.alarm_active_until
            color = (0, 0, 255) if is_breached else (0, 220, 100)  # BGR

            if zg.zone_type == "tripwire" and len(zg.raw_coords) >= 2:
                # Draw tripwire line
                p1 = (int(zg.raw_coords[0][0] * w), int(zg.raw_coords[0][1] * h))
                p2 = (int(zg.raw_coords[1][0] * w), int(zg.raw_coords[1][1] * h))
                cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)
                cv2.circle(frame, p1, 5, color, -1)
                cv2.circle(frame, p2, 5, color, -1)
                # Label
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                cv2.putText(frame, f"TRIPWIRE: {zg.name}", (mid_x - 40, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            elif zg.polygon is not None and len(zg.raw_coords) >= 3:
                # Draw polygon
                pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zg.raw_coords], np.int32)
                pts = pts.reshape((-1, 1, 2))
                # Fill transparent
                fill_color = (0, 0, 180) if is_breached else (0, 180, 60)
                cv2.fillPoly(overlay, [pts], fill_color)
                # Boundary line
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
                # Zone name tag
                label_pt = (pts[0][0][0], max(20, pts[0][0][1] - 8))
                cv2.putText(frame, f"ZONE: {zg.name}", label_pt,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Blend polygon overlay
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # 2. Draw People Tracks
        for t in tracks:
            tid = t["track_id"]
            x1, y1, x2, y2 = map(int, t["bbox"])
            foot_x, foot_y = t["foot_point"]
            trail = t.get("trail", [])

            # Draw motion trail
            if len(trail) > 1:
                for idx in range(1, len(trail)):
                    pt_a = trail[idx - 1]
                    pt_b = trail[idx]
                    cv2.line(frame, pt_a, pt_b, (255, 200, 0), 2, cv2.LINE_AA)

            # Person bounding box (Cyan)
            box_color = (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Bottom footpoint marker
            cv2.circle(frame, (foot_x, foot_y), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (foot_x, foot_y), 9, (255, 255, 255), 1, cv2.LINE_AA)

            # Track ID label
            conf_str = f"#{tid} ({t['confidence']:.2f})"
            (tw, th), _ = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), box_color, -1)
            cv2.putText(frame, conf_str, (x1 + 3, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. HUD (Heads-Up Display)
        # Top banner
        hud_bar_h = 36
        cv2.rectangle(frame, (0, 0), (w, hud_bar_h), (15, 20, 25), -1)
        
        # Red flashing border on intrusion
        if current_time < self.alarm_active_until:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            cv2.putText(frame, "⚠️ INTRUSION ALARM ACTIVE", (w // 2 - 130, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        # Status text
        status_text = f"STREAM: {self.name} | FPS: {self.fps_meter:.1f} | PEOPLE: {self.people_count}"
        cv2.putText(frame, status_text, (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (time_w, _), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, time_str, (w - time_w - 12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    def _handle_event(self, evt: Dict[str, Any], current_frame: np.ndarray):
        """Processes ENTER / EXIT / LOITER events, records snapshots/clips, logs to DB, and alerts."""
        event_type = evt["event_type"]
        track_id = evt["track_id"]
        zone_id = evt["zone_id"]
        zone_name = evt["zone_name"]
        confidence = evt.get("confidence", 0.0)

        # Console print as requested in Phase 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 EVENT: {event_type} | Zone: {zone_name} | Track: #{track_id} | Conf: {confidence:.2f}")

        # Save snapshot
        event_db_id = None
        snapshot_url = None
        clip_url = None
        snapshot_path = None

        db: Session = SessionLocal()
        try:
            db_event = Event(
                stream_id=self.stream_id,
                zone_id=zone_id,
                track_id=track_id,
                event_type=event_type,
                timestamp=datetime.utcnow(),
                confidence=confidence,
                details=evt.get("details", "")
            )
            db.add(db_event)
            db.commit()
            db.refresh(db_event)
            event_db_id = db_event.id

            # Save snapshot on ENTER, LOITER, and CROSS
            if event_type in ["ENTER", "LOITER", "CROSS_IN", "CROSS_OUT"]:
                snapshot_path = ClipRecorder.save_snapshot(current_frame, event_db_id)
                db_event.snapshot_path = snapshot_path
                snapshot_url = f"/api/events/{event_db_id}/snapshot"

                # Trigger asynchronous video clip recording (Pre + Post buffer)
                pre_frames = self.ring_buffer.get_snapshot_frames()
                clip_queue = queue.Queue(maxsize=120)

                with self.subscriber_lock:
                    self.clip_subscribers.append(clip_queue)

                def on_clip_saved(saved_path: str):
                    with self.subscriber_lock:
                        if clip_queue in self.clip_subscribers:
                            self.clip_subscribers.remove(clip_queue)
                    # Update DB with clip path
                    clip_db = SessionLocal()
                    try:
                        ev = clip_db.query(Event).filter(Event.id == event_db_id).first()
                        if ev:
                            ev.clip_path = saved_path
                            clip_db.commit()
                    finally:
                        clip_db.close()

                h, w = current_frame.shape[:2]
                ClipRecorder.record_event_clip_async(
                    event_id=event_db_id,
                    pre_frames=pre_frames,
                    post_buffer_seconds=settings.POST_BUFFER_SECONDS,
                    fps=self.source.fps if self.source else 25.0,
                    width=w,
                    height=h,
                    frame_subscriber_queue=clip_queue,
                    on_complete_callback=on_clip_saved
                )
                clip_url = f"/api/events/{event_db_id}/clip"

            db.commit()
        except Exception as e:
            logger.error(f"Error persisting event to database: {e}")
        finally:
            db.close()

        # Build payload for WebSocket and Alert channels
        payload = {
            "type": "ALERT",
            "event_id": event_db_id or "",
            "stream_id": self.stream_id,
            "stream_name": self.name,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "track_id": track_id,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "snapshot_url": snapshot_url,
            "clip_url": clip_url,
            "message": f"Person #{track_id} {event_type} in {zone_name}"
        }

        # WebSocket broadcast
        if self.ws_broadcast:
            self.ws_broadcast(payload)

        # Dispatch external alerts on alarms
        if event_type in ["ENTER", "LOITER", "CROSS_IN"]:
            # Run async alert dispatch in background thread
            threading.Thread(
                target=self._dispatch_alert_async,
                args=(payload, snapshot_path),
                daemon=True
            ).start()

    def _dispatch_alert_async(self, payload: Dict[str, Any], snapshot_path: Optional[str]):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(AlertManager.dispatch_alert(payload, snapshot_path))
        except Exception as e:
            logger.error(f"Error during alert dispatch: {e}")
        finally:
            loop.close()

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.frame_lock:
            return self.latest_jpeg_frame
