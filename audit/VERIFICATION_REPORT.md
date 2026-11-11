# Audit fixes and verification

11 November 2026. Base revision: `e3e6f30`. This verification includes the AI's existing uncommitted changes plus the completion work in this session. Delivery status is recorded in the Git history for the audit-fix branch.

**Result: 78 regression tests pass. All 20 original audit items were reviewed. Remaining software defects were fixed or unsafe behavior disabled; physical accuracy and deployment validation remain unproven.**

The inherited patch passed 65 checks but left incomplete timing, wire-touch handling, camera calibration, incident transitions, benchmark evidence, and application integration. Eight additional core tests and five application tests now cover these gaps.

## Status of the original findings

| Item | Result |
| --- | --- |
| 1. Streamlit display crash | Inherited width fix verified with real inference, display, evidence, and tables. |
| 2. Timing | Shared FrameClock uses media time/FPS for files and monotonic acquisition time for live feeds. All four applications pass intervals into tracking and monitoring; Kalman transition/noise use the same interval. |
| 3. Occlusion speed | Tracks accumulate every frame interval until the next valid world observation. Variable-interval regression returns 3.6 km/h instead of 21.6 km/h. |
| 4. Tripwire samples | Last off-wire point retained per track/zone. True crossings, touch-and-return, and motion beyond wire endpoints are distinguished by tests. |
| 5. Climbing integration | Configured zones and current projected feet determine proximity, including polygon-only perimeters. No geometry means no assumed proximity. CLI offers --pose; desktop uses installed pose weights. |
| 6. Camera calibration | Calibrator saves source and frame dimensions. Automatic loading requires both to match. Unmatched feeds keep pixel tracking with metric readings/spatial alarms disabled. Explicit legacy calibration remains an operator choice. |
| 7. Overlapping zones | Shared monitor evaluates polygons and wires independently; overlap and two-person tests pass. |
| 8. Confirmation | Inherited prior-confirmation/min-hits guard verified; recovered unconfirmed tracks below threshold return to tentative state. |
| 9. Smoothing | Inherited initial-history/tie fix verified. Threat score and indicators now match the selected smoothed posture. Original failed test passes. |
| 10. Calibration validation | Finite matrices, ordered finite bounds, finite/equal point lists, and complete binding metadata checked. Failed reprojection returns infinity; valid errors use RMS. |
| 11. Network access | Loopback default retained. Remote HTTP requires TRIPWIRE_TOKEN and browser Basic auth as operator. Remote/missing/wrong credentials denied; valid credentials accepted. GET pose route is read-only. |
| 12. Multiple viewers | Whole-frame lock retained; recent JPEG shared between viewers. Two simultaneous callers run one inference. Processing is demand-driven rather than a background capture service. |
| 13. Uploads | Session-owned temporary directory and content-derived paths prevent collisions and stale same-name replacements. Replaced files are removed after capture release; identical reruns do not rewrite. Remaining directory cleanup follows session garbage collection. |
| 14. Incidents | Per-track/zone ENTER, LOITER, EXIT, crossing, speed, and action transitions replace timer-based repeated incidents. Two people each get evidence; sustained presence does not duplicate events. Fabricated certainty removed. |
| 15. Headless CLI | Inherited conditional GUI cleanup verified with both original integration tests and all five complete videos. No cleanup exceptions remain in these runs. |
| 16. MOT scores | Global assignment retained; invalid pairings penalized before optimization so valid match count takes priority. Original ordering probe and adversarial threshold test pass. |
| 17. Benchmarks | Synthetic operations labeled honestly. DIoU outcomes and smoothing come from production code; results regenerated with method/environment metadata. Unsupported headline claims withdrawn. Labeled accuracy remains unestablished. |
| 18. Detector inputs | Streamlit clean-frame split verified; desktop now also infers on an unchanged copy. UI test asserts overlays never change detector input. |
| 19. Desktop failure | Disconnected Q behavior verified with mocked camera/window calls. Loop releases resources in finally; pause does not advance tracking. Native display not exercised. |
| 20. Pose availability | Inherited Docker preloading of both models retained; unavailable pose is reflected in controls/API. Both installed models ran. Docker build remains unexecuted because its engine is stopped. |

## Evidence

- [Final suite XML](verification_final.xml): 78 tests, zero failures.
- [Additional core tests](../tests/test_audit_completion.py) and [application tests](../tests/test_application_flows.py).
- [Original audit probes](test_audit_regressions.py): all pass. Crossing now exercises the stateful monitor; climbing uses an actual configured perimeter.
- [Streamlit smoke](runtime_streamlit.json): initial render, eight real-inference frames, disconnect stop, and Stop control pass. Capture is bounded for the test; YOLO is real.
- [FastAPI smoke](runtime_web.json): dashboard/controls, both detector modes, decodable JPEGs, read-only GET, and first MJPEG route chunk pass. HTTP checks ran outside the sandbox because local socket-pair initialization was restricted inside it.
- [Pose integration](runtime_calibrated.json): **24 real person detections, 23 confirmed-track observations, one ENTER and one LOITER**, and all 24 pose CLI output frames decode. A deliberately broad test zone and explicitly selected legacy mapping exercise integration, not physical accuracy.
- [Full video results](runtime_pipeline.json): all **1,224 output frames decode**, with **no cleanup exceptions**.
- [Regenerated feedback experiments](../evaluation/benchmark_feedback_results.json).

| Clip | Input/output frames | Whole-run processing FPS |
| --- | ---: | ---: |
| cctv_sample.mp4 | 250 / 250 | 9.33 |
| corridor_tripwire_breach.mp4 | 120 / 120 | 11.59 |
| high_speed_fleeing_breach.mp4 | 60 / 60 | 10.21 |
| solutions_ci_demo.mp4 | 62 / 62 | 10.31 |
| walking_cctv.mp4 | 732 / 732 | 10.76 |

Whole-run FPS includes model load, inference, tracking, rendering, and recording. Other audit work overlapped portions of the run. Default unbound samples intentionally ran without metric calibration. These are local observations, not guaranteed throughput or measured accuracy.

The isolated benchmark measured about 98.16 ms YOLO inference and a 99.23 ms sum of separately timed components, excluding ingestion/UI/recording. Actual DIoU acceptance reached 30 tested pixels; abrupt-jump identity retention was false. Earlier hardcoded 80-pixel/100% claims were removed. Simulated smoothing has 27 raw versus 2 smoothed flickers, not a measured CCTV alarm-reduction score.

## Operational changes and limits

Calibrate each camera using the README sidecar convention before expecting metric geometry and spatial alarms. Explicit legacy files are accepted for compatibility only when the operator selects them. Captures reopen with fresh tracking/monitoring state; incident history remains until the session ends. Logs and evidence are bounded session memory, not durable forensic storage.

Remote FastAPI needs TRIPWIRE_TOKEN and HTTPS at the deployment proxy. Streamlit still requires the hosting platform's access boundary before public exposure. The historical portfolio remains as design notes with its old numeric/resume claims prominently marked unverified.

**Not verified:** physical webcams, real RTSP/network failures, interactive calibration clicks, native desktop rendering, long-duration/many-viewer load, TLS/proxy deployment, a fresh Docker build or dependency environment, or labeled detection/pose accuracy. Docker's Linux engine pipe was unavailable. No missing physical calibration or annotations were invented.

Existing AI edits were preserved and extended; global Python packages were not replaced. Local isolated dependencies, generated media, and caches are excluded from Git. Reproducible checks and compact verification evidence are retained.
