What this repo does: TRIP-WIRE detects people in video, tracks their positions, maps their feet onto a ground plane, and shows zone and tripwire alarms beside a top-down radar. It exposes Streamlit, FastAPI/MJPEG, and OpenCV desktop/CLI interfaces. This audit assumes one camera per engine and one operator; the shared web engine was also reviewed for two simultaneous viewers.

Audit date: 11 November 2026, Asia/Calcutta. Source revision: `e3e6f30`. Application source was not changed. Audit scripts, test results, downloaded model weights, and isolated runtime dependencies were added locally.

**Verdict: working computer-vision prototype, not ready for dependable perimeter monitoring or public deployment.** Real inference and video encoding work, but the main Streamlit view crashes in the installed environment, alarm timing and crossings have correctness defects, and published evaluation numbers are not reproducible from this checkout.

## Scope and evidence

Read the application entry points, all seven CV core modules, evaluation scripts, all eight existing test modules, calibration data, Dockerfile, dependency/configuration files, and README. Reviewed the evaluation report and portfolio claims relevant to implemented behavior. No database, account system, durable incident store, CI workflow, or separate alarm dispatcher exists in the tracked tree.

| Check | Result |
| --- | --- |
| Initial `python -m pytest` | Collection blocked: Shapely missing |
| Isolated dependency setup | Shapely 2.2.0 installed under `.audit_deps`; global environment unchanged |
| Existing full suite, after setup | **55 passed, 3 failed**, 58 tests total, 21.29 seconds |
| Extra audit regression probes | **7 failed**, each demonstrating a separate existing defect |
| Python syntax | All 23 original Python files parsed; compile check also passed |
| Input video integrity | All 5 supplied files fully decoded: **1,224 frames**, no count mismatch |
| Full real-video CLI runs | All **1,224 output frames** recorded and decoded correctly; all five runs raised the same headless cleanup exception |
| Streamlit application smoke test | Initial page passed; Start Live ran inference, then crashed on first image display |
| FastAPI application smoke test | Dashboard, feed-switch/pose routes, 8 real frames in each detector mode, and first MJPEG route chunk passed |
| Docker execution | Blocked: Docker Desktop Linux engine pipe absent |

Existing failures: `test_temporal_posture_smoothing_suppresses_transient_glitch`, `test_pipeline_execution_on_synthetic_video`, and `test_pipeline_with_custom_homography_calibration`. The pipeline tests complete frame processing and fail on unconditional GUI cleanup in a headless OpenCV installation.

Evidence files: [existing suite XML](full_tests.xml), [extra regression XML](regressions.xml), [regression source](test_audit_regressions.py), [runtime harness](runtime_checks.py), [video inventory and package versions](runtime_inventory.json), [Streamlit failure](runtime_streamlit.json), and [full video output results](runtime_pipeline.json). FastAPI runtime results are recorded separately in `runtime_web.json`.

| Full clip | Input/output frames | Measured processing FPS |
| --- | ---: | ---: |
| cctv_sample.mp4 | 250 / 250 | 8.21 |
| corridor_tripwire_breach.mp4 | 120 / 120 | 12.14 |
| high_speed_fleeing_breach.mp4 | 60 / 60 | 10.75 |
| solutions_ci_demo.mp4 | 62 / 62 | 10.15 |
| walking_cctv.mp4 | 732 / 732 | 12.02 |

These are local whole-run measurements including model load, inference, tracking, rendering, and writing, not an isolated warmed-up inference benchmark. Other audit work overlapped part of the run. They do not independently establish detection recall or alarm accuracy.

FastAPI evidence: [runtime web checks](runtime_web.json). Both modes produced decodable 1200×500 JPEGs. The eight-frame samples contained no active tracks, so this check establishes executable inference/rendering and route behavior, not detection recall. Standard-mode timing includes first-call startup and is unsuitable for comparing model throughput. FastAPI HTTP checks initially stalled inside the sandbox while Python created an asyncio socket pair; rerunning the same harness outside the sandbox completed successfully. This was an execution restriction, not counted as a project defect.

Environment: Windows, Python 3.13.4, Torch 2.7.1, Ultralytics 8.3.156, Streamlit 1.48.0. Both `opencv-python` 4.12.0.88 and `opencv-python-headless` 4.13.0.92 are installed; imported OpenCV is **4.13.0, GUI NONE**. This environment conflict is relevant to desktop and headless results. Dependency minimums permit the installed Streamlit version.

## Must fix

1. **Start Live crashes on the first displayed frame** — `app_streamlit.py:16–25,584`.
   - **What this is:** One compatibility helper supplies width arguments to several different Streamlit widgets.
   - **Problem:** It inspects `st.button`, then passes `width="stretch"` to `st.image`. In Streamlit 1.48.0 the image call compares that string to zero and raises `TypeError`; reproduced with AppTest and actual YOLO inference.
   - **Fix:** Choose supported arguments for each target widget, or pin and test one compatible Streamlit version. Do not infer every widget's API from the button signature.
   - **If we skip it:** The primary surveillance interface cannot display its live feed in an allowed dependency configuration.

2. **Speed assumes 25 FPS even for the included 12 FPS clips** — `cv_core/tracking/tracker.py:280–285`; tracker construction in every application entry point.
   - **What this is:** Ground displacement is divided by a fixed time step to calculate physical speed.
   - **Problem:** All entry points use the default 0.04-second tracker step. The walking and corridor clips are actually 12 FPS, so equal per-frame displacement is interpreted at about **2.08 times** its source-time speed; live camera delays are also ignored. Streamlit separately adds a fixed 0.05 seconds to dwell time (`app_streamlit.py:442,477`). At 5 processed FPS, a configured 1.5-second dwell needs about 6 real seconds.
   - **Fix:** Use source timestamps for files and monotonic acquisition timestamps for live input; pass elapsed time into tracking and dwell calculations.
   - **If we skip it:** Speed thresholds and time-based alarms change with clip frame rate and machine performance.

3. **Recovering a lost track creates a false speed spike** — `cv_core/tracking/tracker.py:270–285`.
   - **What this is:** World history is updated only when a detection is matched.
   - **Problem:** After missing frames, displacement from the last observation is still divided by one frame interval. The audit's constant 1 m/s motion reports **18 km/h instead of 3.6 km/h** after four missing detections.
   - **Fix:** Store each observation's timestamp or elapsed frame count and divide by the whole interval.
   - **If we skip it:** Ordinary occlusions can trigger speeding and alter pose classification.

4. **A crossing is missed if one observation lands on the wire** — `cv_core/geometry/zones_3d.py:123–129`.
   - **What this is:** The crossing rule requires both adjacent samples to be strictly on opposite sides.
   - **Problem:** The path `(5,1) → (5,0) → (5,-1)` across a horizontal wire at Y=0 produces no event in either step. The failing audit probe reproduces it.
   - **Fix:** Track the last nonzero side per track and wire and emit one crossing when that side changes, retaining finite-segment intersection checks.
   - **If we skip it:** Quantized or slowly moving targets can cross without a tripwire alarm.

5. **Climbing detection is disconnected from the application** — `cv_core/tracking/tracker.py:424`; `cv_core/kinetics/pose_classifier.py:178`.
   - **What this is:** Climbing requires raised hands and `is_near_perimeter=True`.
   - **Problem:** No application caller passes perimeter proximity to classification. Identical raised-hand keypoints classify as climbing when called directly with proximity, but stay standing through the tracker in the audit probe. CLI and desktop additionally load only the standard detector, so their instantiated pose classifier never receives keypoints.
   - **Fix:** Compute proximity from the configured zones and current position and supply it to the existing classifier; use a pose model where pose behavior is advertised.
   - **If we skip it:** The advertised climbing alarm cannot be reached through normal application inference.

6. **Switching cameras reuses an unrelated calibration** — `app_streamlit.py:309–314`; `scripts/web_demo.py:57–63,114–117`; `scripts/demo_gui.py:49–54`.
   - **What this is:** A homography is a mapping tied to a particular camera position and image coordinates.
   - **Problem:** Every feed, uploaded video, and webcam uses the same calibration JSON, despite included resolutions ranging from 640×360 to 768×432. Feed switching resets the tracker but does not select a matching calibration. Missing calibration silently becomes an identity pixel-to-meter mapping.
   - **Fix:** Associate calibration with feed identity and dimensions. Require calibration or visibly disable physical units and metric alarms for unmatched feeds.
   - **If we skip it:** Radar positions and km/h readings can look precise while being physically wrong.

7. **A polygon can hide a tripwire alarm** — `app_streamlit.py:461–487`.
   - **What this is:** Zones are evaluated in a loop that stops after the first containing polygon.
   - **Problem:** The stress preset puts its tripwire at Y=2.2 inside a polygon spanning Y=1.2–2.8. A target crossing inside that polygon hits the polygon branch and `break`, so the wire is never checked; the polygon grace period can hide the immediate crossing.
   - **Fix:** Evaluate wires independently of polygon dwell decisions and collect all applicable reasons before choosing the displayed reason.
   - **If we skip it:** Zone order changes security behavior and suppresses configured tripwires.

8. **Unconfirmed tracks bypass their confirmation threshold** — `cv_core/tracking/tracker.py:626–641`.
   - **What this is:** New tracks should need `min_hits` before becoming confirmed.
   - **Problem:** Every missed tentative track becomes LOST, and any matched LOST track becomes CONFIRMED without checking prior confirmation. With `min_hits=3`, detection → miss → detection already emits a confirmed target; the output condition also hardcodes `hits >= 2` for tentative tracks.
   - **Fix:** Preserve whether a track was previously confirmed, and apply the configured threshold consistently to tentative recovery and output.
   - **If we skip it:** Intermittent false detections defeat the debounce guarantee.

9. **Posture smoothing fails its own regression test** — `cv_core/tracking/tracker.py:193–195,318–335`; `tests/test_feedback_loop_optimizations.py:94`.
   - **What this is:** A short history is meant to suppress a one-frame posture glitch.
   - **Problem:** Initial `standing` plus observations `running,running,standing` forms a tie; oldest insertion order selects standing. Also, the displayed posture is smoothed while the threat score is copied directly from the latest raw classification, so labels and alarm decisions can disagree.
   - **Fix:** Define deterministic tie behavior, avoid treating an invented initial posture as an observation, and derive posture and threat from the same selected state.
   - **If we skip it:** The HUD flickers and transient classifications can alter alarms despite smoothing.

10. **Non-finite calibration matrices are accepted as calibrated** — `cv_core/geometry/homography.py:38–53,89–91,156–165`.
    - **What this is:** Calibration input is validated before it drives every projection.
    - **Problem:** A 3×3 matrix filled with NaNs passes the determinant check and sets `is_calibrated=True`. Separately, reprojection error returns **0.0** when every sample is invalid, incorrectly representing total failure as perfect accuracy. Both are reproduced by audit probes.
    - **Fix:** Reject non-finite matrices and invalid bounds; reject empty, mismatched, or wholly unprojectable validation samples. Label mean distance correctly or compute the documented root-mean-square statistic.
    - **If we skip it:** A broken calibration can silently remove spatial monitoring while presenting a successful calibration state.

11. **The network demo exposes camera video and controls without authorization** — `scripts/web_demo.py:344–380`.
    - **What this is:** The FastAPI service binds to all network interfaces and shares one engine.
    - **Problem:** Any client able to reach port 8000 can read `/stream`, switch feeds, or toggle pose. Even GET `/api/toggle_pose` changes state. This is a local-demo limitation that becomes a security defect when deployed as the README's surveillance service.
    - **Fix:** Default the demo to loopback; require an authenticated boundary before network exposure and remove the state-changing GET route.
    - **If we skip it:** Reachable clients can view surveillance footage and interfere with monitoring.

12. **Two viewers advance and modify the same tracking engine** — `scripts/web_demo.py:127–189,250,363–375`.
    - **What this is:** Every stream subscriber calls `engine.get_frame()` on the same global capture and tracker.
    - **Problem:** The lock ends after capture, before inference and tracker mutation. Two viewers consume different frames, duplicate inference, and can update the tracker out of order. Switching feeds during inference can insert an old-feed detection into the newly reset tracker.
    - **Fix:** Have one producer read/infer/track each camera and publish the latest encoded frame to subscribers. For a single-viewer demo, serialize the entire operation and explicitly enforce that limit.
    - **If we skip it:** Opening a second dashboard can change tracking results and camera processing rate.

13. **Uploads with equal filenames overwrite each other's feeds** — `app_streamlit.py:174–182`.
    - **What this is:** Uploaded videos are written to a shared local directory.
    - **Problem:** Two sessions uploading `clip.mp4` both write `data/videos/uploaded_clip.mp4`; an active capture can see a truncated/replaced file. The same upload is written again on reruns, and there is no deletion lifecycle for uploaded files.
    - **Fix:** Generate a unique server-side path per upload/session, write once, and remove it when no capture needs it.
    - **If we skip it:** Users interfere with each other's video and disk use grows over time.

14. **Incident recording repeats one target and can starve another** — `app_streamlit.py:535–566`.
    - **What this is:** The evidence locker and incident table share a global two-second snapshot throttle.
    - **Problem:** A continuous breach generates another “incident” every two seconds. If the first target in iteration order remains in breach, it takes each available slot and another simultaneous intruder may never get evidence. The log exports the literal `CONFIRMED (98.4%)`, unrelated to model certainty. There is no implemented ENTER/EXIT/LOITER event lifecycle matching the evaluation report.
    - **Fix:** Record events per track/zone transition, throttle evidence per incident, and remove unsupported confidence percentages.
    - **If we skip it:** Counts overstate incidents while missing evidence for other intruders; exported certainty is misleading.

## Should fix

15. **Headless CLI processing raises a GUI error after recording** — `scripts/run_cv_pipeline.py:174`.
    - **What this is:** The `--no-preview` path is intended for non-GUI execution.
    - **Problem:** Cleanup always calls `cv2.destroyAllWindows()`, even when no window was created. Both existing integration tests fail there with the installed headless OpenCV build. This is environment-dependent; it is not proof that every OpenCV installation fails.
    - **Fix:** Call window cleanup only when preview was enabled. Keep one OpenCV distribution in the runtime environment and verify its actual imported build.
    - **If we skip it:** Automated jobs can report failure after otherwise successful recordings.

16. **Tracking accuracy changes when ground-truth rows are reordered** — `evaluation/evaluate_mot.py:67–94`.
    - **What this is:** The evaluator greedily assigns each ground-truth box to its best remaining prediction.
    - **Problem:** The audit's two overlapping targets produce one true positive and MOTA=0 in one input order, versus two true positives and MOTA=1 after reversing the same ground-truth list. The IDF1 path uses a separate global association and remains 1 in that example.
    - **Fix:** Use threshold-aware global per-frame assignment with a documented identity-continuity policy; retain the ordering regression.
    - **If we skip it:** Reported accuracy depends on annotation ordering rather than only tracker quality.

17. **Published performance and reliability claims lack matching evidence** — `cv_core/models/benchmark.py:18–64,95–101`; `scripts/benchmark_models.py:28`; `evaluation/benchmark_feedback_loop.py:152–172,269–295`; `evaluation/EVALUATION_REPORT.md`.
    - **What this is:** Benchmark commands and markdown reports support the project's accuracy/FPS claims.
    - **Problem:** The main profiler computes `sin+cos` and random thresholding, but labels them neural inference and NMS. The DIoU benchmark never runs its proposed baseline tracker, assigns an unused `_linear_assignment`, and hardcodes 100% preservation/80px tolerance; its stored table actually fails at 70/80px. Pose smoothing is reimplemented differently from production. The checked-in real-inference JSON reports **6 FPS**, whereas the README claims 40.1 FPS; hardware/workload provenance is insufficient to reconcile them. No tracked ground-truth annotations or predictions reproduce 96.8% MOTA/94.7% IDF1. The crossing clip has 62 frames locally, versus 150 in the evaluation table.
    - **Fix:** Label synthetic microbenchmarks accurately; measure the actual application and tracker code; publish exact hardware, versions, inputs, annotations, predictions, and generated scores. Remove unsupported fixed success strings.
    - **If we skip it:** Users cannot independently validate performance or decide whether alarms are dependable.

18. **Graphics are drawn onto the detector's input** — `app_streamlit.py:375–410`; `scripts/demo_gui.py:118–126`.
    - **What this is:** Grid lines and zone fills are composited before model inference.
    - **Problem:** Enabling a display option changes the pixels sent to YOLO, including a 25% zone tint and text. The CLI/web paths infer on the original frame, so equivalent clips follow different perception paths. The size of any accuracy effect was not measured.
    - **Fix:** Infer on the original frame and draw only on a separate display copy afterward.
    - **If we skip it:** Presentation controls can change detections and make cross-interface comparisons unreliable.

19. **The desktop demo cannot handle Quit after capture failure** — `scripts/demo_gui.py:93–106,212–215`.
    - **What this is:** OpenCV keyboard events are handled near the end of the processing loop.
    - **Problem:** When a webcam disconnects or a source cannot be read, the loop sleeps and continues before reaching `waitKey`, so it never processes Q or feed-switch keys. This path was verified by code tracing, not physical-camera execution.
    - **Fix:** Process window/key events in the failure branch too, render a disconnected status, and release resources in `finally`.
    - **If we skip it:** Recovering or closing the demo after camera failure can require terminating the process externally.

20. **The image cannot guarantee offline pose startup** — `Dockerfile:36`; `scripts/web_demo.py:43–44`; `app_streamlit.py:45`.
    - **What this is:** The container preloads only standard detector weights.
    - **Problem:** Streamlit pose selection requires downloading an additional model; FastAPI silently disables pose if that file is absent while its toggle can still claim ON. The audit needed a separate pose download and one retry after a network interruption.
    - **Fix:** Bundle both models when offering both modes, or clearly disable unavailable modes and expose their actual state.
    - **If we skip it:** Offline deployments cannot deliver the pose behavior shown in the controls.

## Maintenance and test coverage

The existing core tests usefully cover geometry round trips, Kalman convergence, basic identity recovery, NaN coordinate handling, rendering, and isolated pose rules. The two CLI integration tests use black synthetic frames and only check nonempty output files: they cannot prove person detection, correct alarms, real-time timing, or recorded frame integrity. This audit adds full output-decoding checks and failure probes, but supplies no new labeled security dataset.

There are no existing Streamlit start/stop/export tests, authenticated route tests, multi-viewer tests, camera-specific calibration tests, or physical-speed ground-truth tests. A UI health endpoint alone would not detect the reproduced first-frame crash. The incident store is bounded in memory (8 images, 100 rows); it is not durable evidence storage and clears when session/process state is lost.

The codebase is small enough to keep its existing structure. Reuse `draw_pose_skeleton` in Streamlit instead of maintaining another skeleton implementation (about 25–30 lines can go), delete the unused benchmark baseline/history scaffolding (about 5–10 lines), and remove the tracker-module reload at Streamlit import after verifying rerun identity behavior. No dependency deletion is justified without a clean dependency install and deployment test. Consolidate timing and alarm decisions only after their behavior is defined and tested; a new framework is unnecessary.

## Recommended repair order

1. Restore Streamlit display and headless CLI completion; get all existing tests green.
2. Correct timestamps, occlusion intervals, wire sampling, zone ordering, and confirmation rules. Keep the seven audit reproductions as regression tests.
3. Tie calibration and proximity to each camera; make UI feature claims match active models.
4. Protect network access, serialize camera processing, and isolate uploaded files.
5. Implement incident transitions and honest certainty reporting, then regenerate accuracy/performance reports from labeled inputs.

## Reproduce

From the repository root in PowerShell, after the isolated dependencies and weights used in this audit are present:

```powershell
$env:PYTHONPATH = "$(Join-Path (Get-Location) '.audit_deps');$(Get-Location)"
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) '.audit_runtime'
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest --junitxml=audit/full_tests.xml
python -m pytest audit/test_audit_regressions.py --junitxml=audit/regressions.xml
python audit/runtime_checks.py inventory
python audit/runtime_checks.py streamlit
python audit/runtime_checks.py web
python audit/runtime_checks.py pipeline
```

The regression command intentionally fails until its documented production defects are fixed. The pipeline harness records cleanup exceptions and separately asserts that every expected output frame decodes; output success does not erase the exception. Streamlit uses a bounded real capture wrapper to stop after eight frames; it does not replace YOLO inference. FastAPI uses its test client and the stream route's first generated MJPEG chunk, not a browser/network load test.

**Verdict:** Fix display/startup, timing, missed crossings, calibration, and access control before treating this as a surveillance service.

**Lean:** Approximately 30–40 lines removable through existing skeleton reuse and unused benchmark scaffolding; 0 dependencies proven removable.

**Not checked:** Physical webcams, RTSP/network loss on real hardware, interactive calibration clicks, native desktop rendering, browser visual/accessibility behavior, complete Streamlit incident/export interactions beyond the first-frame crash, sustained multi-viewer load, fresh Docker build, fresh full dependency install, vulnerability database scan, or independently labeled detection/pose accuracy. Portfolio prose was checked for relevant claims, not edited line by line.

**Risk:** Test passes and decodable videos do not establish correct physical calibration or reliable security alarms. The reproduced defects remain unfixed.
