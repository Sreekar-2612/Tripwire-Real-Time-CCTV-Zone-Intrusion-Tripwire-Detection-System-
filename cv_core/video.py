import time

import cv2
import numpy as np


class FrameClock:
    """Use media time for files and acquisition time for live cameras."""

    def __init__(self, capture, live=False, now=time.monotonic):
        self.capture = capture
        self.live = live
        self.now = now
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.interval = 1.0 / fps if np.isfinite(fps) and fps > 0 else .04
        self.previous = now() if live else None

    def tick(self):
        current = self.now() if self.live else self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        dt = current - self.previous if self.previous is not None else self.interval
        self.previous = current
        return float(dt) if np.isfinite(dt) and dt > 0 else self.interval
