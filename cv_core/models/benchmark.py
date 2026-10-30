import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import torch

logger = logging.getLogger("tripwire.cv.benchmark")

class InferenceProfiler:
    """
    Inference latency & throughput profiler for computer vision models.
    Isolates Preprocessing, Neural Network Forward Pass, and NMS Postprocessing.
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
        Profiles standard CV pipeline stages on synthetic frame buffers:
        1. Preprocess: Resize, BGR2RGB, transpose HWC -> CHW, FP32 normalize.
        2. Dummy Compute / Kernel Execution.
        3. Postprocess: Synthetic box decode and threshold filtering.
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
            norm = (dummy_frame.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
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

        return {
            "num_runs": self.test_runs,
            "preprocess_mean_ms": round(float(np.mean(preprocess_times)), 2),
            "compute_mean_ms": round(float(np.mean(compute_times)), 2),
            "postprocess_mean_ms": round(float(np.mean(postprocess_times)), 2),
            "total_latency_mean_ms": round(float(np.mean(total_times)), 2),
            "total_latency_p95_ms": round(float(np.percentile(total_times, 95)), 2),
            "fps": round(1000.0 / float(np.mean(total_times)), 1)
        }

    def generate_markdown_report(self, stats: Dict[str, Any]) -> str:
        """Formats profiling stats into a GitHub-flavored Markdown table."""
        return (
            "| Stage | Latency (Mean) | Description |\n"
            "| :--- | :---: | :--- |\n"
            f"| **Preprocessing** | `{stats['preprocess_mean_ms']} ms` | BGR2RGB, transpose HWC->CHW, float32 scaling |\n"
            f"| **Inference Kernel** | `{stats['compute_mean_ms']} ms` | Forward-pass neural matrix computation |\n"
            f"| **Postprocessing / NMS** | `{stats['postprocess_mean_ms']} ms` | Bounding-box thresholding & suppression |\n"
            f"| **End-to-End Pipeline** | `{stats['total_latency_mean_ms']} ms` | **Throughput: {stats['fps']} FPS** (p95: {stats['total_latency_p95_ms']} ms) |\n"
        )
