import argparse
import sys
import torch
import numpy as np
from cv_core.models.benchmark import InferenceProfiler

def main():
    parser = argparse.ArgumentParser(description="Tripwire CV Pipeline Latency & Throughput Benchmark")
    parser.add_argument("--imgsz", type=int, default=640, help="Input frame dimension (default 640)")
    parser.add_argument("--runs", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    args = parser.parse_args()

    print("="*75)
    print("⚡ TRIPWIRE COMPUTER VISION PERFORMANCE PROFILER")
    print(f"Device: {'CUDA GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU (PyTorch Multithreaded)'}")
    print(f"Frame Size: {args.imgsz}x{args.imgsz} | Iterations: {args.runs} (Warmup: {args.warmup})")
    print("="*75 + "\n")

    profiler = InferenceProfiler(warmup_runs=args.warmup, test_runs=args.runs)
    stats = profiler.profile_dummy_pipeline(imgsz=args.imgsz)

    print(profiler.generate_markdown_report(stats))
    print(f"\nThroughput: {stats['fps']} Frames Per Second")
    print(f"Total Mean Latency: {stats['total_latency_mean_ms']} ms (95th Percentile: {stats['total_latency_p95_ms']} ms)")

if __name__ == "__main__":
    main()
