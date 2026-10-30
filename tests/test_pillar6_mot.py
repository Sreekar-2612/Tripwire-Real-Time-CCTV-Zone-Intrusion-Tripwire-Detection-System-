import pytest
from evaluation.evaluate_mot import MOTEvaluator

def test_mot_evaluator_perfect_tracking():
    """Verify that perfect tracking yields 100% MOTA, MOTP, and IDF1."""
    evaluator = MOTEvaluator(iou_threshold=0.5)

    gt = {
        1: [{"track_id": 1, "bbox": [10.0, 10.0, 50.0, 50.0]}],
        2: [{"track_id": 1, "bbox": [15.0, 10.0, 55.0, 50.0]}],
        3: [{"track_id": 1, "bbox": [20.0, 10.0, 60.0, 50.0]}]
    }
    preds = {
        1: [{"track_id": 101, "bbox": [10.0, 10.0, 50.0, 50.0]}],
        2: [{"track_id": 101, "bbox": [15.0, 10.0, 55.0, 50.0]}],
        3: [{"track_id": 101, "bbox": [20.0, 10.0, 60.0, 50.0]}]
    }

    metrics = evaluator.evaluate_sequence(gt, preds)
    assert metrics["MOTA"] == 1.0
    assert metrics["MOTP"] == 1.0
    assert metrics["IDF1"] == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["id_switches"] == 0

def test_mot_evaluator_id_switch_detection():
    """Verify that identity swap is properly counted as an ID switch."""
    evaluator = MOTEvaluator(iou_threshold=0.5)

    gt = {
        1: [{"track_id": 1, "bbox": [10.0, 10.0, 50.0, 50.0]}],
        2: [{"track_id": 1, "bbox": [15.0, 10.0, 55.0, 50.0]}]
    }
    # Prediction changes ID on frame 2
    preds = {
        1: [{"track_id": 101, "bbox": [10.0, 10.0, 50.0, 50.0]}],
        2: [{"track_id": 999, "bbox": [15.0, 10.0, 55.0, 50.0]}]
    }

    metrics = evaluator.evaluate_sequence(gt, preds)
    assert metrics["id_switches"] == 1
    assert metrics["MOTA"] < 1.0
