# Evaluation evidence

The earlier 40.1 FPS, 96.8% MOTA, 94.7% IDF1, and DeepSORT comparison tables were not reproducible from this repository. They are withdrawn as validated results. The checkout does not contain the labeled annotations, prediction exports, or baseline runs needed to establish those accuracy claims.

Current software verification is recorded in [the fix report](../audit/VERIFICATION_REPORT.md), including 78 passing regressions, all five supplied videos, full decoding of 1,224 output frames, Streamlit/FastAPI smoke tests, and a 24-frame real pose/tracking/incident exercise. Full-video processing ranged from 9.33 to 11.59 FPS locally, including initialization and recording. Those runs intentionally disabled unmatched metric calibration; the separate pose exercise explicitly selected a legacy mapping and broad test zone. Neither establishes physical accuracy or a hardware-independent throughput guarantee.

`scripts/benchmark_models.py` measures synthetic CPU transpose/normalization, sin/cos tensor operations, and random-box filtering. It does not measure YOLO inference or NMS. `evaluation/benchmark_feedback_loop.py` times actual YOLO inference and separate component experiments; their sum excludes ingestion, UI, and recording and is not application throughput.

The regenerated [feedback results](benchmark_feedback_results.json) include method and environment metadata. DIoU measurements now apply the production acceptance gate and report actual track history; hardcoded 80-pixel tolerance/perfect retention claims are gone. Posture smoothing calls production tracking code. The ankle-noise experiment remains a simulation rather than measured camera error reduction.

Publishing MOTA/MOTP/IDF1 or alarm precision/recall requires labeled annotations, camera calibration, model/version/settings, per-frame predictions, exact commands, and baseline results. Validate speed against independently measured distances and timestamps. Synthetic scores cannot substitute for field accuracy.
