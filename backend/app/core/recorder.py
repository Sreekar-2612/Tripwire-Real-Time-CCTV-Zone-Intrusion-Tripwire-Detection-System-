import os
import cv2
import time
import logging
import threading
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from backend.app.config import settings

logger = logging.getLogger("tripwire.recorder")

class FrameRingBuffer:
    """Thread-safe ring buffer storing recent frames for pre-alarm video clips."""
    def __init__(self, max_seconds: float = 3.0, fps: float = 25.0):
        self.capacity = max(10, int(max_seconds * fps))
        self.buffer = deque(maxlen=self.capacity)
        self.lock = threading.Lock()

    def append(self, frame: np.ndarray, timestamp: float):
        with self.lock:
            # Store a copy so modifications to ongoing frame don't mutate buffer
            self.buffer.append((frame.copy(), timestamp))

    def get_snapshot_frames(self) -> List[Tuple[np.ndarray, float]]:
        with self.lock:
            return list(self.buffer)


class ClipRecorder:
    """Handles snapshot creation and asynchronous pre/post-alarm MP4 clip recording."""
    def __init__(self):
        pass

    @staticmethod
    def save_snapshot(frame: np.ndarray, event_id: str) -> str:
        """Saves annotated frame as JPEG snapshot. Returns filesystem path."""
        filename = f"{event_id}.jpg"
        filepath = settings.SNAPSHOTS_DIR / filename
        try:
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            logger.info(f"Saved snapshot: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save snapshot {filepath}: {e}")
            return ""

    @staticmethod
    def record_event_clip_async(
        event_id: str,
        pre_frames: List[Tuple[np.ndarray, float]],
        post_buffer_seconds: float,
        fps: float,
        width: int,
        height: int,
        frame_subscriber_queue,
        on_complete_callback=None
    ):
        """Spawns background thread to collect post-frames and write the full MP4 clip."""
        thread = threading.Thread(
            target=ClipRecorder._clip_worker,
            args=(event_id, pre_frames, post_buffer_seconds, fps, width, height, frame_subscriber_queue, on_complete_callback),
            daemon=True
        )
        thread.start()

    @staticmethod
    def _clip_worker(
        event_id: str,
        pre_frames: List[Tuple[np.ndarray, float]],
        post_buffer_seconds: float,
        fps: float,
        width: int,
        height: int,
        frame_queue,
        on_complete_callback
    ):
        clip_filename = f"{event_id}.mp4"
        clip_path = settings.CLIPS_DIR / clip_filename
        
        all_frames = [f[0] for f in pre_frames]
        target_post_frames = max(10, int(post_buffer_seconds * fps))
        start_time = time.time()

        # Collect subsequent post-event frames
        collected_post = 0
        timeout = post_buffer_seconds + 3.0  # safe margin

        while collected_post < target_post_frames and (time.time() - start_time) < timeout:
            try:
                frame = frame_queue.get(timeout=0.2)
                if frame is not None:
                    all_frames.append(frame.copy())
                    collected_post += 1
            except Exception:
                continue

        if not all_frames:
            logger.warning(f"No frames to write for clip {clip_path}")
            return

        # Write to MP4 using OpenCV VideoWriter
        # Try 'avc1' first for web browser compatibility, fallback to 'mp4v'
        fourcc_options = ['avc1', 'mp4v', 'XVID']
        writer = None

        for code in fourcc_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*code)
                test_writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
                if test_writer.isOpened():
                    writer = test_writer
                    break
            except Exception:
                continue

        if writer is None or not writer.isOpened():
            logger.error(f"Failed to initialize VideoWriter for {clip_path}")
            return

        try:
            for frm in all_frames:
                # Ensure correct frame dimensions
                if frm.shape[1] != width or frm.shape[0] != height:
                    frm = cv2.resize(frm, (width, height))
                writer.write(frm)
            writer.release()
            logger.info(f"Successfully recorded intrusion clip: {clip_path} ({len(all_frames)} frames)")

            if on_complete_callback:
                on_complete_callback(str(clip_path))
        except Exception as e:
            logger.error(f"Error encoding video clip {clip_path}: {e}")
            if writer:
                writer.release()
