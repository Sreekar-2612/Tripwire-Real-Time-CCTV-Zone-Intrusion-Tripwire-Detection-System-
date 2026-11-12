import time
import json
import inspect
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

# Try importing streamlit
try:
    import streamlit as st
except ImportError:
    st = None

def get_width_kwargs(stretch: bool = True):
    """Provides version-compatible kwargs for BUTTONS only across Streamlit versions."""
    try:
        if st is not None and "width" in inspect.signature(st.button).parameters:
            return {"width": "stretch" if stretch else "content"}
    except Exception:
        pass
    return {"use_container_width": stretch}

WIDTH_KWARGS = get_width_kwargs()

# Fix #1: st.image does NOT accept 'width="stretch"' — always use use_container_width.
IMAGE_KWARGS = {"use_container_width": True}
# Fix #1: st.dataframe also uses use_container_width, not width=.
DATAFRAME_KWARGS = {"use_container_width": True}

from ultralytics import YOLO
from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D
from cv_core.tracking.tracker import MultiTargetTracker
from cv_core.kinetics.pose_classifier import PoseKineticClassifier, draw_pose_skeleton
from cv_core.visualizer.bev_radar import BEVRadarRenderer
from cv_core.video import FrameClock
from cv_core.monitoring import ZoneMonitor
import hashlib
import tempfile

@st.cache_resource
def get_object_detector():
    """Caches standard YOLOv8 nano person detector in memory."""
    return YOLO("yolov8n.pt")

@st.cache_resource
def get_pose_detector():
    """Caches YOLOv8 pose detector in memory for kinetic threat classification."""
    return YOLO("yolov8n-pose.pt")

def main():
    if st is None:
        print("Please install Streamlit using: pip install streamlit")
        return

    st.set_page_config(
        page_title="Tripwire — Cyber SOC Spatial Intelligence HUD",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Session State Initialization
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "current_feed" not in st.session_state:
        st.session_state.current_feed = None
    if "tracker" not in st.session_state:
        st.session_state.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
    if "zone_monitor" not in st.session_state:
        st.session_state.zone_monitor = ZoneMonitor()
    if "incident_log" not in st.session_state:
        st.session_state.incident_log = []
    if "evidence_locker" not in st.session_state:
        st.session_state.evidence_locker = []

    # Custom Cyber SOC Dark Styling
    st.markdown("""
        <style>
        .stApp { background-color: #0b0f19; color: #f1f5f9; }
        .main-header { font-size: 26px; font-weight: 800; color: #38bdf8; margin-bottom: 2px; letter-spacing: -0.5px; }
        .sub-header { font-size: 13px; color: #94a3b8; margin-bottom: 16px; }
        .soc-badge {
            display: inline-block; padding: 4px 10px; border-radius: 9999px;
            font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
            margin-right: 8px;
        }
        .badge-live { background-color: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
        .badge-dlt { background-color: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }
        .badge-kalman { background-color: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid #a855f7; }
        .stMetric { background-color: #111827; padding: 12px; border-radius: 8px; border: 1px solid #1f2937; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🛡️ TRIP-WIRE: Cyber SOC Spatial Intelligence & BEV Radar HUD</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="sub-header">
            <span class="soc-badge badge-live">● SURVEILLANCE DEMO</span>
            <span class="soc-badge badge-dlt">CAMERA CALIBRATION REQUIRED</span>
            <span class="soc-badge badge-kalman">4D KALMAN TRACKING</span>
            <span>Real-Time Planar Homography, Constant-Velocity State Estimation & Threat Kinetics</span>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Surveillance & Model Controls")

    model_architecture = st.sidebar.selectbox(
        "AI Vision Architecture",
        [
            "YOLOv8-Nano (Person Detection & Tracking)",
            "YOLOv8-Pose (17-Joint Kinetic Threat & Action Recognition)"
        ],
        index=0,
        help="YOLOv8-Nano offers ultra-fast object tracking; YOLOv8-Pose adds 17 COCO keypoint kinetics for climbing, crawling, and fall classification."
    )
    use_pose = "Pose" in model_architecture

    feed_option = st.sidebar.selectbox(
        "Select Video Feed",
        [
            "Template: Corridor Tripwire Crossing (Breach Demo)",
            "Template: High-Speed Fleeing Intruder (Alarm Demo)",
            "Walking Corridor CCTV (Full 60s)",
            "Hallway Intrusion",
            "Pedestrian Crossing",
            "Live Laptop Camera (Webcam 0)",
            "External Camera (Webcam 1)",
            "Upload Custom MP4"
        ]
    )

    video_path = None
    if feed_option == "Template: Corridor Tripwire Crossing (Breach Demo)":
        video_path = "data/videos/corridor_tripwire_breach.mp4"
    elif feed_option == "Template: High-Speed Fleeing Intruder (Alarm Demo)":
        video_path = "data/videos/high_speed_fleeing_breach.mp4"
    elif feed_option == "Walking Corridor CCTV (Full 60s)":
        video_path = "data/videos/walking_cctv.mp4"
    elif feed_option == "Hallway Intrusion":
        video_path = "data/videos/cctv_sample.mp4"
    elif feed_option == "Pedestrian Crossing":
        video_path = "data/videos/solutions_ci_demo.mp4"
    elif feed_option == "Live Laptop Camera (Webcam 0)":
        video_path = 0
    elif feed_option == "External Camera (Webcam 1)":
        video_path = 1
    elif feed_option == "Upload Custom MP4":
        uploaded_file = st.sidebar.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
        if uploaded_file:
            if '_upload_dir' not in st.session_state:
                st.session_state._upload_dir = tempfile.TemporaryDirectory(prefix='tripwire_upload_')
            content = uploaded_file.getbuffer()
            digest = hashlib.sha256(content).hexdigest()
            temp_path = Path(st.session_state._upload_dir.name) / (digest + Path(uploaded_file.name).suffix.lower())
            previous_upload = st.session_state.get('_upload_path')
            if previous_upload and previous_upload != str(temp_path):
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                Path(previous_upload).unlink(missing_ok=True)
            if not temp_path.exists():
                temp_path.write_bytes(content)
            st.session_state._upload_path = str(temp_path)
            video_path = str(temp_path)

    conf_thresh = st.sidebar.slider("Detection Confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    infer_imgsz = st.sidebar.select_slider(
        "Inference Resolution (imgsz)",
        options=[384, 480, 640],
        value=480,
        help="Lower resolutions reduce compute cost; validate accuracy for your camera."
    )
    speed_alarm = st.sidebar.slider("Speed Alarm Threshold (km/h)", min_value=1.0, max_value=25.0, value=5.0, step=0.5)
    dwell_confirm_sec = st.sidebar.slider(
        "Breach Dwell / Debounce (sec)",
        min_value=0.0,
        max_value=4.0,
        value=1.5,
        step=0.5,
        help="Target must linger inside zone for this duration to confirm breach. Prevents false alarms during normal transit."
    )
    zone_preset = st.sidebar.radio(
        "Security Zone Layout",
        ["Realistic Security (Side Vault & Checkpoint)", "Demo Stress Test (Center Hallway)"],
        index=0,
        help="Realistic layout monitors off-limits doors/alcoves. Stress test places the zone in the center walkway."
    )
    show_grid = st.sidebar.checkbox("Project Perspective Ground Grid", value=True)
    use_demo_calib = st.sidebar.checkbox(
        "Apply Floor Calibration (Enable Velocity & Radar)",
        value=True,
        help="Applies default calibrated floor homography to estimate ground coordinates, metric velocity (km/h), and BEV radar blips when camera-specific calibration is not loaded."
    )

    st.sidebar.markdown("---")
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("▶️ Start Live", type="primary", key="btn_start_live", **WIDTH_KWARGS):
            st.session_state.is_running = True
            st.rerun()
    with col_btn2:
        if st.button("⏹️ Stop Feed", key="btn_stop_feed", **WIDTH_KWARGS):
            st.session_state.is_running = False
            if st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                except Exception:
                    pass
                st.session_state.cap = None
            st.session_state.current_feed = None
            st.rerun()

    if st.session_state.incident_log:
        st.sidebar.markdown("---")
        with st.sidebar.expander(f"📥 Export Incident Logs ({len(st.session_state.incident_log)})", expanded=False):
            df_sb = pd.DataFrame(st.session_state.incident_log[::-1])
            csv_sb = df_sb.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv_sb, "tripwire_incident_audit.csv", "text/csv", key="sb_download_csv", **WIDTH_KWARGS)
            json_sb = json.dumps(st.session_state.incident_log, indent=2).encode('utf-8')
            st.download_button("📥 Download JSON", json_sb, "tripwire_forensic_report.json", "application/json", key="sb_download_json", **WIDTH_KWARGS)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**CV Algorithms Active:**")
    st.sidebar.markdown("- `ByteTrack` Two-Stage Bipartite Association")
    st.sidebar.markdown("- `4D Kalman Filter` (Joseph Form Stabilized)")
    st.sidebar.markdown("- `17-Joint COCO` Kinetic Threat Classifier")
    st.sidebar.markdown("- `Hartley-Normalized` DLT Homography")
    st.sidebar.markdown("- `BEV Radar Map` (20m x 20m Ground Plane)")

    # Dynamic Top Metric Banners
    col1, col2, col3, col4 = st.columns(4)
    metric_conf = col1.empty()
    metric_fps = col2.empty()
    metric_tracks = col3.empty()
    metric_breaches = col4.empty()

    metric_conf.metric("Vision Architecture", "YOLOv8-Pose" if use_pose else "YOLOv8-Nano", "Active Model")
    metric_fps.metric("Processing Speed", "Standby", "0.0 FPS")
    metric_tracks.metric("Active Targets", "0", "Awaiting Feed")
    metric_breaches.metric("Intrusion Breaches", "0", "Secure")

    st.markdown("---")

    # Main Display Area
    dual_view_placeholder = st.empty()
    status_text = st.empty()

    # Evidence Locker and Incident Table Placeholders
    evidence_placeholder = st.empty()
    table_placeholder = st.empty()

    calibration_upload = st.sidebar.file_uploader('Calibration JSON for this feed', type=['json'])

    # Handle Feed Switching
    feed_changed = (st.session_state.current_feed != video_path)
    if feed_changed:
        try:
            if st.session_state.cap is not None:
                st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None
        st.session_state.current_feed = video_path
        st.session_state.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
        st.session_state.zone_monitor = ZoneMonitor()

    if st.session_state.is_running and video_path is not None:
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            if isinstance(video_path, int):
                cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(video_path, cv2.CAP_MSMF)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(video_path)
            else:
                cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                if isinstance(video_path, int):
                    status_text.error(
                        f"❌ Could not access webcam device index #{video_path}.\n\n"
                        "• Ensure camera is connected and recognized in Windows Device Manager.\n"
                        "• Check Windows Settings > Privacy & security > Camera to ensure apps can access your camera.\n"
                        "• Ensure another application (Zoom, Teams, Discord, Chrome) is not currently locking the webcam."
                    )
                else:
                    status_text.error(f"❌ Could not open video file ({video_path}). Please verify the file exists.")
                st.session_state.is_running = False
                return

            st.session_state.cap = cap
            st.session_state.current_feed = video_path
            st.session_state.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
            st.session_state.zone_monitor = ZoneMonitor()

        cap = st.session_state.cap
        clock = FrameClock(cap, live=isinstance(video_path, int) or '://' in str(video_path))
        detector = get_pose_detector() if use_pose else get_object_detector()
        pose_classifier = PoseKineticClassifier(keypoint_conf_threshold=0.35) if use_pose else None
        tracker = st.session_state.tracker
        radar = BEVRadarRenderer(radar_size=480, world_extent_m=20.0, grid_step_m=2.0)
        
        image_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        calibration_path = None
        if calibration_upload is not None:
            if '_calibration_dir' not in st.session_state:
                st.session_state._calibration_dir = tempfile.TemporaryDirectory(prefix='tripwire_calibration_')
            calibration_path = Path(st.session_state._calibration_dir.name) / 'calibration.json'
            calibration_path.write_bytes(calibration_upload.getbuffer())
        try:
            transformer = HomographyTransformer.for_source(video_path, image_size, calibration_path)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            status_text.error(f'Invalid calibration: {exc}')
            st.session_state.is_running = False
            cap.release()
            st.session_state.cap = None
            return
        if not transformer.is_calibrated and use_demo_calib and Path("data/calibration.json").exists():
            transformer = HomographyTransformer.load_calibration("data/calibration.json")
        if not transformer.is_calibrated:
            st.warning('This feed is uncalibrated. Ground positions, metric speed, and spatial alarms are disabled. Select a calibration JSON for this camera or check "Apply Floor Calibration".')
        calibration_key = (str(video_path), transformer.H.tobytes() if transformer.is_calibrated else None, zone_preset, use_pose)
        if st.session_state.get('_calibration_key') != calibration_key:
            st.session_state.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
            st.session_state.zone_monitor = ZoneMonitor()
            st.session_state._calibration_key = calibration_key
        tracker = st.session_state.tracker

        if zone_preset == "Demo Stress Test (Center Hallway)":
            zones = [
                MetricZone3D("z1", "RESTRICTED VAULT (CENTER HALL)", "polygon", [(3.5, 1.2), (7.5, 1.2), (7.5, 2.8), (3.5, 2.8)]),
                MetricZone3D("tw1", "PERIMETER FENCE", "tripwire", [(2.0, 2.2), (8.0, 2.2)])
            ]
        else:  # Realistic Security
            if "Pedestrian" in feed_option:
                zones = [
                    MetricZone3D("z_ped", "RESTRICTED PLAZA BUFFER", "polygon", [(4.0, 8.0), (16.0, 8.0), (16.0, 14.0), (4.0, 14.0)]),
                    MetricZone3D("tw_ped", "CROSSWALK TRIPWIRE", "tripwire", [(2.0, 11.0), (18.0, 11.0)])
                ]
            elif "Laptop" in feed_option or "Webcam" in feed_option:
                zones = [
                    MetricZone3D("z_webcam", "RESTRICTED WORKSPACE", "polygon", [(2.0, 1.0), (8.0, 1.0), (8.0, 4.0), (2.0, 4.0)]),
                    MetricZone3D("tw_webcam", "DESK TRIPWIRE", "tripwire", [(1.5, 2.5), (8.5, 2.5)])
                ]
            elif "High-Speed" in feed_option:
                zones = [
                    MetricZone3D("z_flee", "RESTRICTED VAULT", "polygon", [(5.5, 2.0), (9.0, 2.0), (9.0, 6.0), (5.5, 6.0)]),
                    MetricZone3D("tw_flee", "PERIMETER TRIPWIRE", "tripwire", [(1.5, 3.8), (8.5, 3.8)])
                ]
            else:
                zones = [
                    MetricZone3D("z_alcove", "RESTRICTED OFFICE VAULT", "polygon", [(6.0, 3.0), (9.5, 3.0), (9.5, 7.5), (6.0, 7.5)]),
                    MetricZone3D("tw_chk", "CHECKPOINT TRIPWIRE", "tripwire", [(2.0, 4.5), (8.5, 4.5)])
                ]

        frame_count = 0
        t0 = time.time()

        try:
            # Continuous processing loop
            while st.session_state.is_running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                except Exception:
                    ret, frame = False, None

                if not ret or frame is None:
                    if isinstance(video_path, str) and not video_path.isdigit():
                        try:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
                            st.session_state.tracker = tracker
                            st.session_state.zone_monitor = ZoneMonitor()
                            clock = FrameClock(cap)
                            ret, frame = cap.read()
                        except Exception:
                            ret, frame = False, None

                    if not ret or frame is None:
                        status_text.error(f"⚠️ Video stream or camera feed '{feed_option}' was disconnected or ended.")
                        st.session_state.is_running = False
                        if st.session_state.cap is not None:
                            try:
                                st.session_state.cap.release()
                            except Exception:
                                pass
                            st.session_state.cap = None
                        break

                frame_elapsed_dt = clock.tick()
                frame_count += 1
                t_frame_start = time.time()

                # Fix #18: run inference on the ORIGINAL clean frame, then draw
                # grid/zones/annotations only on a display copy.
                display_frame = frame.copy()

                # Grid & Security Zone Projection on Camera View
                if transformer.is_calibrated:
                    if show_grid:
                        segments = transformer.project_ground_grid_to_image(display_frame.shape[:2], grid_step_m=2.0)
                        grid_overlay = display_frame.copy()
                        for pt1, pt2 in segments:
                            cv2.line(grid_overlay, pt1, pt2, (60, 80, 90), 1, cv2.LINE_AA)
                        cv2.addWeighted(grid_overlay, 0.4, display_frame, 0.6, 0, display_frame)

                    for z in zones:
                        if z.zone_type == "polygon":
                            try:
                                pts = [transformer.world_to_image(x, y) for x, y in z.world_coords]
                                pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                                overlay = display_frame.copy()
                                cv2.fillPoly(overlay, [pts_np], (0, 140, 255))
                                cv2.addWeighted(overlay, 0.25, display_frame, 0.75, 0, display_frame)
                                cv2.polylines(display_frame, [pts_np], True, (0, 165, 255), 2, cv2.LINE_AA)
                                cv2.putText(display_frame, z.name, (int(pts[3][0]) + 10, int(pts[3][1]) + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 2, cv2.LINE_AA)
                            except Exception:
                                pass
                        elif z.zone_type == "tripwire":
                            try:
                                p1 = transformer.world_to_image(*z.world_coords[0])
                                p2 = transformer.world_to_image(*z.world_coords[1])
                                cv2.line(display_frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2, cv2.LINE_AA)
                                cv2.putText(display_frame, z.name, (int(p1[0]) + 15, int(p1[1]) - 6),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
                            except Exception:
                                pass

                # Deep Visual Perception — always infer on the ORIGINAL clean frame (fix #18)
                t_infer_start = time.time()
                with torch.inference_mode():
                    results = detector(frame, classes=[0], conf=min(conf_thresh, tracker.low_conf_threshold), imgsz=infer_imgsz, verbose=False)
                t_infer_end = time.time()
                infer_ms = (t_infer_end - t_infer_start) * 1000.0

                detections = []
                for r in results:
                    if r.boxes is not None:
                        for idx_b, box in enumerate(r.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            det_dict = {
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": conf,
                                "foot_point": ((x1 + x2) / 2.0, float(y2))
                            }
                            if use_pose and getattr(r, "keypoints", None) is not None and idx_b < len(r.keypoints):
                                det_dict["keypoints"] = r.keypoints[idx_b].data[0].cpu().numpy()
                            detections.append(det_dict)

                # Tracking with Two-Stage ByteTrack Association & Kinetics
                t_track_start = time.time()
                active_tracks = tracker.update(
                    detections,
                    homography_transformer=transformer,
                    pose_classifier=pose_classifier,
                    zones=zones,                   # Fix #5: pass zones for proximity computation
                    elapsed_dt=frame_elapsed_dt    # Fix #2: pass actual source-FPS interval
                )
                t_track_end = time.time()
                track_ms = (t_track_end - t_track_start) * 1000.0

                radar_tracks = []
                breach_count = 0
                now_ts = time.time()
                alarm_states, frame_events = st.session_state.zone_monitor.update(
                    active_tracks, zones, frame_elapsed_dt, dwell_confirm_sec, speed_alarm)
                for event in frame_events:
                    st.session_state.incident_log.append({
                        'Timestamp': time.strftime('%H:%M:%S'), 'Track ID': f"#{event['track_id']}",
                        'Event': event['event'], 'Zone': event['zone_id'], 'Incident Reason': event['reason']})
                st.session_state.incident_log = st.session_state.incident_log[-100:]

                for trk in active_tracks:
                    tid = trk["track_id"]
                    w_pos = trk.get("world_pos")
                    alarm = alarm_states[tid]
                    is_confirmed_breach = alarm['is_breaching']
                    is_monitoring = alarm['is_monitoring']
                    breach_reasons = alarm['reasons']
                    speed_kmh = float(trk.get('speed', 0))
                    threat_level = trk.get('threat_level', 1)
                    posture_label = trk.get('posture', 'unknown')
                    if use_pose and trk.get('keypoints') is not None:
                        draw_pose_skeleton(display_frame, trk['keypoints'])

                    if is_confirmed_breach:
                        breach_count += 1

                    radar_tracks.append({
                        "track_id": trk["track_id"],
                        "world_pos": w_pos,
                        "world_trail": trk.get("world_trail", []),
                        "speed_kmh": speed_kmh,
                        "vx": trk.get("vx", 0.0),
                        "vy": trk.get("vy", 0.0),
                        "is_breaching": is_confirmed_breach
                    })

                    x1, y1, x2, y2 = map(int, trk["bbox"])
                    if is_confirmed_breach:
                        color = (0, 0, 255)      # Red: Confirmed Intrusion Breach
                        tag_prefix = "🚨"
                    elif is_monitoring:
                        color = (0, 215, 255)    # Amber: Monitoring / Grace Period
                        tag_prefix = "🟡"
                    else:
                        color = (0, 255, 0)      # Green: Authorized Target
                        tag_prefix = "🟢"

                    # Fix #18: draw bounding box and labels on display_frame, not on the YOLO input
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                    speed_label = f"{speed_kmh:.1f}km/h" if w_pos is not None else "speed unavailable"
                    tag_label = f"{tag_prefix} #{trk['track_id']} | {posture_label.upper()} | {speed_label}"
                    if breach_reasons:
                        tag_label += f" | {breach_reasons[0]}"
                    cv2.putText(display_frame, tag_label, (x1, max(16, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

                    new_alarm = next((e for e in frame_events if e['track_id'] == tid and
                                      e['event'] in ('LOITER', 'CROSS_IN', 'CROSS_OUT', 'SPEED_START', 'ACTION_START')), None)
                    if new_alarm is not None:
                        ih, iw = frame.shape[:2]
                        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(iw, x2), min(ih, y2)
                        if cx2 > cx1 and cy2 > cy1:
                            st.session_state.evidence_locker.append({
                                'ts': now_ts, 'time_str': time.strftime('%H:%M:%S'), 'track_id': tid,
                                'action': posture_label.upper(), 'threat': threat_level,
                                'reason': new_alarm['reason'], 'speed': speed_label,
                                'image': cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)})
                            st.session_state.evidence_locker = st.session_state.evidence_locker[-8:]

                # BEV Radar Rendering
                radar_img = radar.render(zones if transformer.is_calibrated else [], radar_tracks,
                                         title="GROUND RADAR" if transformer.is_calibrated else "UNCALIBRATED - METRIC ALARMS DISABLED")
                # Fix #18: compose with display_frame (annotated copy), not the raw inference frame
                dual_view = BEVRadarRenderer.create_dual_view(display_frame, radar_img)

                dual_view_resized = cv2.resize(dual_view, (1280, 480), interpolation=cv2.INTER_AREA)
                dual_view_rgb = cv2.cvtColor(dual_view_resized, cv2.COLOR_BGR2RGB)
                dual_view_placeholder.image(dual_view_rgb, channels="RGB", **IMAGE_KWARGS)

                elapsed = time.time() - t0
                fps = frame_count / max(0.001, elapsed)

                # Dynamically update live metrics
                metric_conf.metric("Vision Architecture", "YOLOv8-Pose" if use_pose else "YOLOv8-Nano", f"Infer: {infer_ms:.1f}ms")
                metric_fps.metric("Processing Speed", f"{fps:.1f} FPS", f"Track: {track_ms:.1f}ms")
                metric_tracks.metric("Active Targets", f"{len(active_tracks)}", f"{len(active_tracks)} in camera frame")
                metric_breaches.metric("Intrusion Breaches", f"{breach_count}", "ALARM ACTIVE" if breach_count > 0 else "Secure")

                status_text.info(f"⚡ Pipeline FPS: **{fps:.1f}** | Inference: **{infer_ms:.1f}ms** | Tracking: **{track_ms:.1f}ms** | Active Targets: **{len(active_tracks)}** | Breaches: **{breach_count}** | Feed: **{feed_option}**")

                # Render Incident Audit Log and Evidence Locker (Live updates)
                if st.session_state.evidence_locker:
                    with evidence_placeholder.container():
                        st.markdown("#### 📸 Intruder Incident Snapshot Evidence Locker")
                        cols = st.columns(min(4, len(st.session_state.evidence_locker)))
                        for idx_snap, snap in enumerate(reversed(st.session_state.evidence_locker[-4:])):
                            with cols[idx_snap % 4]:
                                st.image(snap["image"], caption=f"ID #{snap['track_id']} | {snap['time_str']} | {snap['action']}\n{snap['reason']}", **IMAGE_KWARGS)

                if st.session_state.incident_log:
                    with table_placeholder.container():
                        st.markdown(f"#### 📋 Real-Time Incident Audit Log ({len(st.session_state.incident_log)} Events Recorded)")
                        df_log = pd.DataFrame(st.session_state.incident_log[::-1])
                        st.dataframe(df_log, height=180, **DATAFRAME_KWARGS)
                        st.caption("ℹ️ *To export CSV or JSON forensics reports, click **⏹️ Stop Feed** in the sidebar or expand **📥 Export Incident Logs**.*")

                time.sleep(0.015)

        finally:
            cap.release()
            st.session_state.cap = None

    elif not st.session_state.is_running:
        status_text.warning("⏸️ Surveillance pipeline is paused. Click **▶️ Start Live** in the sidebar to run.")
        if st.session_state.evidence_locker:
            with evidence_placeholder.container():
                st.markdown("#### 📸 Intruder Incident Snapshot Evidence Locker")
                cols = st.columns(min(4, len(st.session_state.evidence_locker)))
                for idx_snap, snap in enumerate(reversed(st.session_state.evidence_locker[-4:])):
                    with cols[idx_snap % 4]:
                        st.image(snap["image"], caption=f"ID #{snap['track_id']} | {snap['time_str']} | {snap['action']}\n{snap['reason']}", **IMAGE_KWARGS)

        if st.session_state.incident_log:
            with table_placeholder.container():
                st.markdown(f"#### 📋 Real-Time Incident Audit Log ({len(st.session_state.incident_log)} Total Events)")
                df_log = pd.DataFrame(st.session_state.incident_log[::-1])
                st.dataframe(df_log, height=180, **DATAFRAME_KWARGS)
                col_csv, col_json = st.columns(2)
                with col_csv:
                    csv_data = df_log.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Export Audit Log (CSV)", csv_data, "tripwire_incident_audit.csv", "text/csv", key="paused_export_csv", **WIDTH_KWARGS)
                with col_json:
                    json_data = json.dumps(st.session_state.incident_log, indent=2).encode('utf-8')
                    st.download_button("📥 Export Forensic Report (JSON)", json_data, "tripwire_forensic_report.json", "application/json", key="paused_export_json", **WIDTH_KWARGS)

if __name__ == "__main__":
    main()
