import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import torch

logger = logging.getLogger("tripwire.cv.benchmark")

class InferenceProfiler:
    """
    Synthetic latency microbenchmark for CV preprocessing and tensor operations.

    Fix #17: This profiler measures preprocessing (transpose/normalise) and
    generic tensor kernel throughput on SYNTHETIC frames.  It does NOT run
    YOLO or any other neural detector.  Results cannot be compared with
    real-inference FPS reported elsewhere in the project.  Use actual video processing timings for representative throughput numbers.
    """
    def __init__(self, warmup_runs: int = 5, test_runs: int = 20):
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs

    def profile_dummy_pipeline(
        self,
        imgsz: int = 640,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Microbenchmarks standard CV preprocessing stages on a synthetic frame:
        1. Preprocess: Transpose HWC -> CHW and FP32 normalise.
        2. Tensor Ops: sin+cos kernel (synthetic placeholder — NOT neural inference).
        3. Postprocess: Threshold filtering on random boxes (NOT real NMS).

        NOTE: Stage 2 timings are NOT neural-network forward-pass timings.
        """
        dummy_frame = np.random.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)

        # Warmup
        for _ in range(self.warmup_runs):
            _ = dummy_frame.astype(np.float32) / 255.0

        preprocess_times = []
        compute_times = []
        postprocess_times = []
        total_times = []

        for _ in range(self.test_runs):
            t0 = time.perf_counter_ns()
            
            # Stage 1: Preprocess
            t_pre_start = time.perf_counter_ns()
            chw = (dummy_frame.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            norm = np.repeat(chw, max(1, batch_size), axis=0) if batch_size > 1 else chw
            t_pre_end = time.perf_counter_ns()
            preprocess_times.append((t_pre_end - t_pre_start) / 1e6)  # ms

            # Stage 2: Tensor Compute
            t_comp_start = time.perf_counter_ns()
            tensor = torch.from_numpy(norm)
            out = torch.sin(tensor) + torch.cos(tensor)  # Representative kernel ops
            _ = out.numpy()
            t_comp_end = time.perf_counter_ns()
            compute_times.append((t_comp_end - t_comp_start) / 1e6)

            # Stage 3: Postprocess / Filter
            t_post_start = time.perf_counter_ns()
            dummy_boxes = np.random.rand(100, 4)
            dummy_scores = np.random.rand(100)
            valid = dummy_scores > 0.4
            _ = dummy_boxes[valid]
            t_post_end = time.perf_counter_ns()
            postprocess_times.append((t_post_end - t_post_start) / 1e6)

            t_total_end = time.perf_counter_ns()
            total_times.append((t_total_end - t0) / 1e6)

        if not total_times:
            return {
                "num_runs": 0,
                "preprocess_mean_ms": 0.0,
                "compute_mean_ms": 0.0,
                "postprocess_mean_ms": 0.0,
                "total_latency_mean_ms": 0.0,
                "total_latency_p95_ms": 0.0,
                "fps": 0.0
            }

        mean_total = float(np.mean(total_times))
        fps_val = round(1000.0 / mean_total, 1) if mean_total > 1e-4 else 0.0

        return {
            "num_runs": self.test_runs,
            "preprocess_mean_ms": round(float(np.mean(preprocess_times)), 2) if preprocess_times else 0.0,
            "compute_mean_ms": round(float(np.mean(compute_times)), 2) if compute_times else 0.0,
            "postprocess_mean_ms": round(float(np.mean(postprocess_times)), 2) if postprocess_times else 0.0,
            "total_latency_mean_ms": round(mean_total, 2),
            "total_latency_p95_ms": round(float(np.percentile(total_times, 95)), 2),
            "fps": fps_val
        }

    def generate_markdown_report(self, stats: Dict[str, Any]) -> str:
        """Formats profiling stats into a GitHub-flavored Markdown table."""
        return (
            "| Stage | Latency (Mean) | Description |\n"
            "| :--- | :---: | :--- |\n"
            f"| **Preprocessing** | `{stats['preprocess_mean_ms']} ms` | Transpose HWC->CHW, float32 scaling (synthetic input) |\n"
            f"| **Synthetic Tensor Operations** | `{stats['compute_mean_ms']} ms` | sin + cos, not neural inference |\n"
            f"| **Random Box Filtering** | `{stats['postprocess_mean_ms']} ms` | Random score thresholding, not NMS |\n"
            f"| **Synthetic Pipeline** | `{stats['total_latency_mean_ms']} ms` | **Synthetic throughput: {stats['fps']} FPS** (p95: {stats['total_latency_p95_ms']} ms) |\n"
        )
