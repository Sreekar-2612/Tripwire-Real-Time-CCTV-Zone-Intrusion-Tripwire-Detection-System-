import pytest
import numpy as np
import tempfile
import cv2
from pathlib import Path

from scripts.run_cv_pipeline import run_cv_engine

def test_pipeline_execution_on_synthetic_video():
    """Verify that run_cv_engine processes a synthetic video file without error."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in_path = Path(tmp_in.name)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
        tmp_out_path = Path(tmp_out.name)

    try:
        # Create a small 10-frame synthetic video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(tmp_in_path), fourcc, 10.0, (640, 480))
        for _ in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # Run pipeline in headless mode
        run_cv_engine(
            video_source=str(tmp_in_path),
            calibration_file=None,
            output_video=str(tmp_out_path),
            show_window=False
        )

        assert tmp_out_path.exists()
        assert tmp_out_path.stat().st_size > 0
    finally:
        if tmp_in_path.exists():
            tmp_in_path.unlink()
        if tmp_out_path.exists():
            tmp_out_path.unlink()

def test_pipeline_with_custom_homography_calibration():
    """Verify run_cv_engine correctly executes with an explicit calibration JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
        tmp_vid_path = Path(tmp_vid.name)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_calib:
        tmp_calib_path = Path(tmp_calib.name)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
        tmp_out_path = Path(tmp_out.name)

    try:
        # Create small synthetic video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(tmp_vid_path), fourcc, 10.0, (320, 240))
        for _ in range(5):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()

        # Write dummy calibration JSON
        import json
        calib_data = {
            "homography_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "world_bounds": [0.0, 0.0, 20.0, 20.0],
            "unit": "meters"
        }
        with open(tmp_calib_path, "w") as f:
            json.dump(calib_data, f)

        # Run pipeline with calibration
        run_cv_engine(
            video_source=str(tmp_vid_path),
            calibration_file=str(tmp_calib_path),
            output_video=str(tmp_out_path),
            show_window=False
        )

        assert tmp_out_path.exists()
        assert tmp_out_path.stat().st_size > 0
    finally:
        if tmp_vid_path.exists():
            tmp_vid_path.unlink()
        if tmp_calib_path.exists():
            tmp_calib_path.unlink()
        if tmp_out_path.exists():
            tmp_out_path.unlink()

