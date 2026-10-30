import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

class MOTEvaluator:
    """
    Standard Computer Vision Multi-Object Tracking (MOT) Metric Evaluator.
    Computes:
    - MOTA (Multiple Object Tracking Accuracy): 1 - (FN + FP + IDSW) / GT
    - MOTP (Multiple Object Tracking Precision): Average overlap/spatial distance
    - IDF1 (Identification F1-score): 2*IDTP / (2*IDTP + IDFP + IDFN)
    - IDSW: Total identity switches
    - False Positive (FP) and False Negative (FN) counts
    """
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    @staticmethod
    def calculate_box_iou(box_a: List[float], box_b: List[float]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def evaluate_sequence(
        self,
        ground_truth: Dict[int, List[Dict[str, Any]]],
        tracker_predictions: Dict[int, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        ground_truth: frame_id -> list of {'track_id': int, 'bbox': [x1, y1, x2, y2]}
        tracker_predictions: frame_id -> list of {'track_id': int, 'bbox': [x1, y1, x2, y2]}
        """
        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        iou_sum = 0.0

        # Mapping gt_id -> last assigned pred_id
        active_associations = {}

        all_frame_ids = sorted(set(list(ground_truth.keys()) + list(tracker_predictions.keys())))

        for f_id in all_frame_ids:
            gt_boxes = ground_truth.get(f_id, [])
            pred_boxes = tracker_predictions.get(f_id, [])

            total_gt += len(gt_boxes)
            matched_pred = set()
            matched_gt = set()

            # Greedy bipartite matching per frame
            for g_idx, gt in enumerate(gt_boxes):
                best_iou = 0.0
                best_p_idx = -1
                for p_idx, pred in enumerate(pred_boxes):
                    if p_idx in matched_pred:
                        continue
                    iou = self.calculate_box_iou(gt["bbox"], pred["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_p_idx = p_idx

                if best_iou >= self.iou_threshold and best_p_idx >= 0:
                    matched_gt.add(g_idx)
                    matched_pred.add(best_p_idx)
                    total_tp += 1
                    iou_sum += best_iou

                    pred_tid = pred_boxes[best_p_idx]["track_id"]
                    gt_tid = gt["track_id"]

                    if gt_tid in active_associations and active_associations[gt_tid] != pred_tid:
                        total_idsw += 1
                    active_associations[gt_tid] = pred_tid

            # Unmatched GT = False Negatives
            total_fn += (len(gt_boxes) - len(matched_gt))
            # Unmatched Pred = False Positives
            total_fp += (len(pred_boxes) - len(matched_pred))

        # Metric Calculations
        mota = (1.0 - ((total_fn + total_fp + total_idsw) / total_gt)) if total_gt > 0 else 1.0
        motp = (iou_sum / total_tp) if total_tp > 0 else 0.0
        
        # True Trajectory IDF1 formulation via Global Bipartite Matching
        gt_ids = sorted(list({gt["track_id"] for f in ground_truth.values() for gt in f}))
        pred_ids = sorted(list({p["track_id"] for f in tracker_predictions.values() for p in f}))

        if gt_ids and pred_ids:
            cost_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int32)
            for g_idx, g_id in enumerate(gt_ids):
                for p_idx, p_id in enumerate(pred_ids):
                    cost_matrix[g_idx, p_idx] = sum(
                        1 for f in all_frame_ids
                        if any(g["track_id"] == g_id and p["track_id"] == p_id and 
                               self.calculate_box_iou(g["bbox"], p["bbox"]) >= self.iou_threshold
                               for g in ground_truth.get(f, []) for p in tracker_predictions.get(f, []))
                    )

            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-cost_matrix)
            idtp = int(sum(cost_matrix[r, c] for r, c in zip(row_ind, col_ind)))
            total_pred = sum(len(p) for p in tracker_predictions.values())
            idfp = int(total_pred - idtp)
            idfn = int(total_gt - idtp)
            id_denom = 2 * idtp + idfp + idfn
            idf1 = (2.0 * idtp / id_denom) if id_denom > 0 else 0.0
        else:
            idf1 = 0.0

        return {
            "total_frames": len(all_frame_ids),
            "total_ground_truth": total_gt,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "id_switches": total_idsw,
            "MOTA": round(mota, 4),
            "MOTP": round(motp, 4),
            "IDF1": round(idf1, 4)
        }

    @staticmethod
    def generate_evaluation_markdown(results: Dict[str, Any], sequence_name: str = "Surveillance-Test") -> str:
        return (
            f"### Multi-Object Tracking Benchmark Results: {sequence_name}\n\n"
            f"| Metric | Score | Industry Standard Description |\n"
            f"| :--- | :---: | :--- |\n"
            f"| **MOTA** (Tracking Accuracy) | **{results['MOTA'] * 100:.2f}%** | 1 - (FN + FP + IDSW) / Total_GT |\n"
            f"| **MOTP** (Spatial Precision) | **{results['MOTP'] * 100:.2f}%** | Average Bounding Box IoU Overlap |\n"
            f"| **IDF1** (Identity F1-Score) | **{results['IDF1'] * 100:.2f}%** | Trajectory ID association stability |\n"
            f"| **False Positives (FP)** | `{results['false_positives']}` | Ghost / transient background detections |\n"
            f"| **False Negatives (FN)** | `{results['false_negatives']}` | Occluded / missed intruder detections |\n"
            f"| **ID Switches (IDSW)** | `{results['id_switches']}` | Track identity swapping errors |\n"
        )

if __name__ == "__main__":
    print("MOTEvaluator benchmark engine ready.")
