import pytest
from cv_core.models.benchmark import InferenceProfiler

def test_inference_profiler_execution():
    profiler = InferenceProfiler(warmup_runs=2, test_runs=5)
    stats = profiler.profile_dummy_pipeline(imgsz=320)

    assert "preprocess_mean_ms" in stats
    assert "compute_mean_ms" in stats
    assert "postprocess_mean_ms" in stats
    assert "total_latency_mean_ms" in stats
    assert stats["fps"] > 0.0

    report = profiler.generate_markdown_report(stats)
    assert "| **Preprocessing** |" in report
    assert "| **End-to-End Pipeline** |" in report
