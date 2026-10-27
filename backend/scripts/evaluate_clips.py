import os
import sys
import time
import cv2
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from backend.app.core.detector import PersonDetectorTracker
from backend.app.core.zones import ZoneManager, ZoneGeometry

def evaluate_clip(
    video_path: str,
    clip_name: str,
    zone_coords: list,
    ground_truth_enter_times: list,  # in seconds
    max_frames: int = 150
):
    print(f"\n--- Evaluating Clip: {clip_name} ---")
    print(f"Path: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open {video_path}")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"Resolution: {w}x{h} @ {fps:.1f} FPS")

    detector = PersonDetectorTracker(confidence=0.35, imgsz=640)
    zm = ZoneManager()
    zone = ZoneGeometry("eval_z1", "Evaluation Zone", "polygon", zone_coords, armed=True)
    zm.zones = {"eval_z1": zone}

    detected_enters = []
    frame_idx = 0
    t_start = time.time()

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        current_time = frame_idx / fps
        tracks = detector.track(frame)
        events = zm.process_tracks(tracks, current_time, debounce_frames=2, cooldown_sec=4.0)

        for e in events:
            if e["event_type"] == "ENTER":
                detected_enters.append(current_time)
                print(f"  Frame {frame_idx:03d} ({current_time:.2f}s) -> ENTER triggered (Track #{e['track_id']}, Conf: {e['confidence']:.2f})")

        frame_idx += 1

    total_time = time.time() - t_start
    measured_fps = frame_idx / max_time if (max_time := max(0.001, total_time)) else 0.0
    cap.release()

    # Calculate Precision and Recall against ground truth
    # Tolerance window of +/- 1.5 seconds
    tolerance = 1.5
    matched_gt = set()
    true_positives = 0
    false_positives = 0

    for det_t in detected_enters:
        matched = False
        for i, gt_t in enumerate(ground_truth_enter_times):
            if abs(det_t - gt_t) <= tolerance and i not in matched_gt:
                matched_gt.add(i)
                true_positives += 1
                matched = True
                break
        if not matched:
            false_positives += 1

    false_negatives = len(ground_truth_enter_times) - len(matched_gt)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0

    result = {
        "clip": clip_name,
        "frames_processed": frame_idx,
        "measured_fps": round(measured_fps, 1),
        "ground_truth_enters": len(ground_truth_enter_times),
        "detected_enters": len(detected_enters),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3)
    }

    print(f"Results for {clip_name}:")
    print(f"  FPS: {result['measured_fps']} | Precision: {result['precision']} | Recall: {result['recall']} | F1: {result['f1_score']}")
    return result

def main():
    recordings_dir = Path(__file__).resolve().parent.parent / "data" / "recordings"

    clips = [
        {
            "name": "Solutions Demo (Pedestrians/Crossing)",
            "path": str(recordings_dir / "solutions_ci_demo.mp4"),
            "zone": [[0.1, 0.2], [0.85, 0.2], [0.85, 0.95], [0.1, 0.95]],
            "ground_truth": [0.8, 2.5]
        },
        {
            "name": "Corridor Walking CCTV",
            "path": str(recordings_dir / "walking_cctv.mp4"),
            "zone": [[0.2, 0.3], [0.8, 0.3], [0.8, 0.95], [0.2, 0.95]],
            "ground_truth": [1.2]
        }
    ]

    results = []
    for c in clips:
        if os.path.exists(c["path"]):
            res = evaluate_clip(c["path"], c["name"], c["zone"], c["ground_truth"])
            if res:
                results.append(res)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY TABLE (CPU INFERENCE & METRICS)")
    print("=" * 70)
    print(f"{'Clip Name':<35} | {'FPS':<6} | {'Precision':<10} | {'Recall':<8} | {'F1':<6}")
    print("-" * 70)
    for r in results:
        print(f"{r['clip']:<35} | {r['measured_fps']:<6} | {r['precision']:<10} | {r['recall']:<8} | {r['f1_score']:<6}")
    print("=" * 70)

if __name__ == "__main__":
    main()
