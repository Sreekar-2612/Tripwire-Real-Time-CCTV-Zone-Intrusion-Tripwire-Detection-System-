import cv2
import time
import logging
import threading
from typing import Optional, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger("tripwire.sources")

class BaseVideoSource(ABC):
    """Abstract base class for all video sources."""
    def __init__(self, source_id: str, source_url: str):
        self.source_id = source_id
        self.source_url = source_url
        self.width = 1280
        self.height = 720
        self.fps = 25.0
        self.is_running = False
        self.is_connected = False
        self.last_frame = None
        self.lock = threading.Lock()

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[any]]:
        pass

    @abstractmethod
    def release(self):
        pass


class OpenCVVideoSource(BaseVideoSource):
    """Handles RTSP, Webcams (numeric index), and recorded video files."""
    def __init__(
        self,
        source_id: str,
        source_url: str,
        source_type: str = "file",
        loop: bool = True
    ):
        super().__init__(source_id, source_url)
        self.source_type = source_type.lower()
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_interval = 1.0 / 25.0
        self.last_read_time = 0.0
        self.reconnect_delay = 2.0
        self.max_reconnect_delay = 30.0

        self._open_capture()

    def _open_capture(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        logger.info(f"Connecting to source '{self.source_url}' ({self.source_type})...")
        
        # Check if source is webcam index
        if self.source_type == "webcam" or self.source_url.isdigit():
            cap_target = int(self.source_url)
            self.cap = cv2.VideoCapture(cap_target, cv2.CAP_DSHOW if cv2.__name__ == 'cv2' else cv2.CAP_ANY)
        else:
            # RTSP or File path
            self.cap = cv2.VideoCapture(self.source_url)

        if not self.cap.isOpened():
            logger.warning(f"Could not open video source: {self.source_url}")
            self.is_connected = False
            return False

        # Read camera metadata
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)

        self.width = w if w > 0 else 1280
        self.height = h if h > 0 else 720
        self.fps = f if (f and f > 5 and f < 120) else 25.0
        self.frame_interval = 1.0 / self.fps
        self.is_connected = True
        logger.info(f"Source connected: {self.source_url} ({self.width}x{self.height} @ {self.fps:.1f}fps)")
        return True

    def read_frame(self) -> Tuple[bool, Optional[any]]:
        """Reads next frame. Handles file looping, pacing, and RTSP reconnect."""
        if not self.is_connected or self.cap is None or not self.cap.isOpened():
            # Try reconnecting for live streams
            if self.source_type in ["rtsp", "webcam"]:
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 1.5, self.max_reconnect_delay)
                if self._open_capture():
                    self.reconnect_delay = 2.0
            return False, None

        # For file sources, pace reading to real-world playback speed
        if self.source_type == "file":
            now = time.time()
            elapsed = now - self.last_read_time
            if elapsed < self.frame_interval:
                time.sleep(max(0.001, self.frame_interval - elapsed))
            self.last_read_time = time.time()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self.source_type == "file" and self.loop:
                # Rewind video to start
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True, frame
            elif self.source_type in ["rtsp", "webcam"]:
                logger.warning(f"Connection lost on {self.source_url}. Reconnecting...")
                self.is_connected = False
                return False, None
            else:
                # End of file without loop
                return False, None

        return True, frame

    def release(self):
        self.is_connected = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.debug(f"Error releasing capture: {e}")
            self.cap = None
        logger.info(f"Released source {self.source_url}")
