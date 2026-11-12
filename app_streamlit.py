import time
import json
import inspect
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

# Try importing streamlit
try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError:
    st = None
    components = None

def get_width_kwargs(stretch: bool = True):
    """Provides version-compatible kwargs for BUTTONS only across Streamlit versions."""
    try:
        if st is not None and "width" in inspect.signature(st.button).parameters:
            return {"width": "stretch" if stretch else "content"}
    except Exception:
        pass
    return {"use_container_width": stretch}

WIDTH_KWARGS = get_width_kwargs()

# st.image and st.dataframe use use_container_width
IMAGE_KWARGS = {"use_container_width": True}
DATAFRAME_KWARGS = {"use_container_width": True}

from ultralytics import YOLO
from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D, get_default_zones
from cv_core.tracking.tracker import MultiTargetTracker
from cv_core.kinetics.pose_classifier import PoseKineticClassifier, draw_pose_skeleton
from cv_core.visualizer.bev_radar import BEVRadarRenderer
from cv_core.video import FrameClock
from cv_core.monitoring import ZoneMonitor
from cv_core.audio.siren import SirenAlertController
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

def open_capture_source(source, max_retries: int = 1):
    """
    Fast, resilient multi-backend video and webcam capture initialization.
    For webcams:
      - Tries requested index with DirectShow (cv2.CAP_DSHOW) on Windows, followed by default backend.
      - If requested index fails, probes adjacent index (1 if 0, 0 if 1).
      - Retains and returns opened camera handle without premature timeout discards.
      - Sets 640x480 resolution.
    For video files:
      - Opens file with standard cv2.VideoCapture(source).
    Returns:
      (cap, actual_source, backend_name, error_message)
    """
    is_webcam = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
    if not is_webcam:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap, source, "File", None
        return None, source, "File", f"Could not open video file ({source}). Please verify the file exists."

    req_idx = int(source)
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Default", None)
    ] if sys.platform.startswith("win") else [("Default", None)]

    # Check requested index first, then adjacent device index
    candidate_indices = [req_idx]
    alt_idx = 1 if req_idx == 0 else 0
    candidate_indices.append(alt_idx)

    for idx in candidate_indices:
        for b_name, b_flag in backends:
            try:
                cap = cv2.VideoCapture(idx, b_flag) if b_flag is not None else cv2.VideoCapture(idx)
            except Exception:
                continue
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                except Exception:
                    pass
                # Quick warmup frame check (doesn't discard opened camera)
                for _ in range(3):
                    try:
                        ret, f = cap.read()
                        if ret and f is not None and f.size > 0:
                            break
                    except Exception:
                        pass
                    time.sleep(0.04)
                return cap, idx, b_name, None
            else:
                try:
                    cap.release()
                except Exception:
                    pass

    err = (
        f"Could not access webcam device index #{req_idx}.\n\n"
        "• Ensure camera is connected and recognized in Windows Device Manager.\n"
        "• Check Windows Settings > Privacy & security > Camera to ensure apps can access your camera.\n"
        "• Ensure another application (Zoom, Teams, Discord, WhatsApp, Chrome) is not currently locking the webcam.\n"
        "• Remote / Multi-Laptop Tip: If viewing from another laptop, select 'Browser Web Camera' in the sidebar to stream directly from your current device's browser!"
    )
    return None, req_idx, None, err

def run_browser_camera_pipeline(
    feed_option, zone_preset, use_pose, use_demo_calib, imgsz, conf_threshold,
    speed_alarm, dwell_confirm_sec, metric_status, metric_fps, metric_tracks,
    metric_threat, metric_breaches, alert_banner_placeholder, dual_view_placeholder,
    viewport_header_placeholder, key="browser_cam_client"
):
    """Universal browser camera frame processing pipeline for local, multi-laptop, and cloud deployments."""
    metric_status.metric("Stream Status", "Browser Camera", "Client HTML5")
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.85); border: 1.5px solid rgba(2, 132, 199, 0.35); border-radius: 16px; padding: 14px 20px; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.03);">
            <div style="font-family: 'Hosta', 'Host Grotesk', sans-serif; font-weight: 800; font-size: 14px; color: #0284c7; letter-spacing: 0.5px;">UNIVERSAL CLIENT-SIDE BROWSER CAMERA</div>
            <div style="font-size: 12px; color: #475569; margin-top: 3px;">
                Captures live video frames directly from your laptop or mobile camera using native browser HTML5. Works across all devices, operating systems, and remote network connections without requiring server-side hardware access.
            </div>
        </div>
    """, unsafe_allow_html=True)
    cam_file = st.camera_input("Security Camera Feed (Client Browser HTML5)", key=key)
    if cam_file is not None:
        raw_bytes = cam_file.getvalue()
        frame = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            detector = get_pose_detector() if use_pose else get_object_detector()
            pose_classifier = PoseKineticClassifier(keypoint_conf_threshold=0.35) if use_pose else None
            tracker = st.session_state.tracker
            radar = BEVRadarRenderer(radar_size=480, world_extent_m=20.0, grid_step_m=2.0)
            image_size = (frame.shape[1], frame.shape[0])
            transformer = HomographyTransformer.for_source(0, image_size)
            if not transformer.is_calibrated and use_demo_calib and Path("data/calibration.json").exists():
                transformer = HomographyTransformer.load_calibration("data/calibration.json")
            zones = get_default_zones(feed_option, preset=zone_preset)
            t_frame_start = time.time()
            display_frame = frame.copy()

            t_infer_start = time.time()
            try:
                results = detector(display_frame, imgsz=imgsz, conf=conf_threshold, verbose=False)
            except Exception:
                results = []
            infer_ms = (time.time() - t_infer_start) * 1000.0

            raw_detections = []
            if results and len(results) > 0:
                r = results[0]
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for idx, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        if cls_id != 0:
                            continue
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        keypoints_data = None
                        if use_pose and hasattr(r, 'keypoints') and r.keypoints is not None:
                            try:
                                kpts = r.keypoints.data[idx].cpu().numpy()
                                if len(kpts) == 17:
                                    keypoints_data = kpts
                            except Exception:
                                keypoints_data = None
                        raw_detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'keypoints': keypoints_data})

            t_track_start = time.time()
            active_tracks = tracker.update(
                raw_detections,
                homography_transformer=transformer if transformer.is_calibrated else None,
                pose_classifier=pose_classifier,
                zones=zones,
                elapsed_dt=0.033
            )
            track_ms = (time.time() - t_track_start) * 1000.0

            now_ts = time.time()
            frame_events = st.session_state.zone_monitor.update(
                active_tracks, zones if transformer.is_calibrated else [], now_ts,
                speed_alarm_kmh=speed_alarm, dwell_sec=dwell_confirm_sec
            )
            alarm_states = st.session_state.zone_monitor.get_alarm_states()

            radar_tracks = []
            breach_count = 0
            for trk in active_tracks:
                tid = trk["track_id"]
                w_pos = trk.get("world_pos")
                alarm = alarm_states.get(tid, {'is_breaching': False, 'is_monitoring': False, 'reasons': []})
                is_confirmed_breach = alarm['is_breaching']
                is_monitoring = alarm['is_monitoring']
                speed_kmh = float(trk.get('speed', 0))
                threat_level = trk.get('threat_level', 1)
                posture_label = trk.get('posture', 'standing')
                if use_pose and trk.get('keypoints') is not None:
                    draw_pose_skeleton(display_frame, trk['keypoints'])
                if is_confirmed_breach:
                    breach_count += 1
                radar_tracks.append({
                    "track_id": trk["track_id"], "world_pos": w_pos,
                    "world_trail": trk.get("world_trail", []), "speed_kmh": speed_kmh,
                    "vx": trk.get("vx", 0.0), "vy": trk.get("vy", 0.0), "is_breaching": is_confirmed_breach
                })
                x1, y1, x2, y2 = map(int, trk["bbox"])
                color = (0, 0, 255) if is_confirmed_breach else ((0, 215, 255) if is_monitoring else (0, 255, 0))
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                speed_label = f"{speed_kmh:.1f}km/h" if w_pos is not None else "uncalibrated"
                tag_label = f"#{trk['track_id']} | {posture_label.upper()} | {speed_label}"
                (tw, th), baseline = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                tag_y = max(th + 6, y1 - 8)
                cv2.rectangle(display_frame, (x1, tag_y - th - 4), (x1 + tw + 6, tag_y + baseline), (15, 20, 25), -1)
                cv2.rectangle(display_frame, (x1, tag_y - th - 4), (x1 + tw + 6, tag_y + baseline), color, 1)
                cv2.putText(display_frame, tag_label, (x1 + 3, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

            radar_img = radar.render(zones if transformer.is_calibrated else [], radar_tracks,
                                     title="GROUND RADAR" if transformer.is_calibrated else "UNCALIBRATED")
            dual_view = BEVRadarRenderer.create_dual_view(display_frame, radar_img)
            dual_view_resized = cv2.resize(dual_view, (1280, 480), interpolation=cv2.INTER_AREA)
            dual_view_rgb = cv2.cvtColor(dual_view_resized, cv2.COLOR_BGR2RGB)

            viewport_header_placeholder.markdown("""
                <div class="viewport-hud-topbar">
                    <div class="viewport-hud-title">
                        <span>PRIMARY OPTICAL FOV (LEFT) // BIRD'S EYE VIEW RADAR 20m (RIGHT)</span>
                    </div>
                    <div class="viewport-hud-badge">LIVE BROWSER CAMERA</div>
                </div>
            """, unsafe_allow_html=True)
            dual_view_placeholder.image(dual_view_rgb, **IMAGE_KWARGS)
            metric_fps.metric("Inference", f"{infer_ms:.0f}ms", f"Track: {track_ms:.0f}ms")
            metric_tracks.metric("Active Targets", str(len(active_tracks)), "Client FOV")
            metric_breaches.metric("Intrusion Breaches", str(breach_count), f"{len(frame_events)} Events")
            if breach_count > 0:
                metric_threat.metric("Threat Index", "CRITICAL", "DEFCON 1 // BREACH")
                alert_banner_placeholder.markdown("""
                    <div class="soc-alert-banner">
                        <span class="soc-alert-icon"></span> CRITICAL PERIMETER BREACH ACTIVE — INTRUDER DETECTED IN RESTRICTED ZONE
                    </div>
                """, unsafe_allow_html=True)
            else:
                metric_threat.metric("Threat Index", "Nominal", "DEFCON 4 // Secure")
                alert_banner_placeholder.empty()

def main():
    if st is None:
        print("Please install Streamlit using: pip install streamlit")
        return

    st.set_page_config(
        page_title="TRIP-WIRE — Spatial Intelligence HUD",
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
    if "siren_controller" not in st.session_state:
        st.session_state.siren_controller = SirenAlertController(enabled=True)

    # Pristine High-End Light Glassmorphism Style Injection
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Host+Grotesk:wght@600;700;800&display=swap');

        :root {
            --glass-bg-base: #f5f7fb;
            --glass-card-bg: rgba(255, 255, 255, 0.76);
            --glass-card-border: rgba(255, 255, 255, 0.85);
            --glass-card-shadow: 0 20px 40px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.03), inset 0 1px 1px rgba(255, 255, 255, 0.95);
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --accent-charcoal: #0f172a;
            --accent-blue: #2563eb;
            --accent-emerald: #059669;
            --accent-crimson: #dc2626;
        }

        /* Strict Global Typography: Arial BOLD by Default */
        body, .stApp, p, span, div, label, input, button, select, textarea, td, th {
            font-family: Arial, Helvetica, sans-serif !important;
            font-weight: bold !important;
        }

        /* Material Symbols Icons must preserve their icon font ligature mapping */
        [data-testid="stIconMaterial"],
        .material-symbols-rounded,
        [class*="material-symbols"] {
            font-family: 'Material Symbols Rounded', sans-serif !important;
            font-weight: normal !important;
        }

        /* Strict Typography: Titles & Headers use 'Hosta' / 'Host Grotesk' BOLD */
        h1, h2, h3, h4, h5, h6,
        .soc-main-heading,
        .soc-main-heading *,
        .sidebar-brand-title,
        .sidebar-brand-title *,
        .sidebar-section-header,
        .sidebar-section-header *,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] *,
        .viewport-hud-title,
        .viewport-hud-title *,
        .section-title,
        .section-title * {
            font-family: 'Hosta', 'Host Grotesk', sans-serif !important;
            font-weight: 700 !important;
        }

        /* Main Canvas Setup */
        .stApp {
            background-color: var(--glass-bg-base) !important;
            color: var(--text-primary) !important;
            min-height: 100vh;
        }
        [data-testid="stAppViewContainer"] {
            background-color: var(--glass-bg-base) !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            position: relative;
            z-index: 1;
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 95% !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(245, 247, 251, 0.45) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
        }

        /* Clean Header: Suppress local Community Cloud Deploy button artifacts */
        .stDeployButton,
        [data-testid="stDeployButton"],
        [data-testid="stToolbarActions"] button:has([data-testid="stIconDeploy"]),
        header button:has([data-testid="stIconDeploy"]),
        div:has(> button:has([data-testid="stIconDeploy"])) {
            display: none !important;
        }

        /* Floating 3D Spheres & Matrix Grid Keyframe Motion */
        @keyframes floatSpheres {
            0% {
                transform: translate3d(0, 0, 0) scale(1);
            }
            33% {
                transform: translate3d(16px, -22px, 0) scale(1.02);
            }
            66% {
                transform: translate3d(-18px, 14px, 0) scale(0.98);
            }
            100% {
                transform: translate3d(0, 0, 0) scale(1);
            }
        }
        @keyframes floatSpheresGrid {
            0% {
                transform: translate3d(0, 0, 0);
            }
            50% {
                transform: translate3d(-10px, -15px, 0);
            }
            100% {
                transform: translate3d(0, 0, 0);
            }
        }

        /* Soft Floating 3D White Translucent Glass Spheres / Orbs */
        .stApp::before {
            content: "";
            position: fixed;
            top: -6%; left: -6%; width: 112vw; height: 112vh;
            background:
                radial-gradient(circle at 14% 18%, rgba(255, 255, 255, 0.98) 0%, rgba(240, 244, 252, 0.85) 40%, rgba(220, 228, 242, 0.35) 65%, transparent 78%),
                radial-gradient(circle at 82% 16%, rgba(255, 255, 255, 0.96) 0%, rgba(238, 243, 251, 0.8) 45%, rgba(218, 226, 240, 0.35) 70%, transparent 82%),
                radial-gradient(circle at 52% 48%, rgba(255, 255, 255, 0.92) 0%, rgba(235, 241, 250, 0.75) 45%, rgba(215, 224, 238, 0.3) 70%, transparent 84%),
                radial-gradient(circle at 88% 82%, rgba(255, 255, 255, 0.98) 0%, rgba(236, 242, 252, 0.85) 40%, rgba(218, 227, 242, 0.35) 68%, transparent 80%),
                radial-gradient(circle at 12% 78%, rgba(255, 255, 255, 0.95) 0%, rgba(238, 243, 251, 0.8) 45%, rgba(216, 225, 239, 0.35) 70%, transparent 82%);
            background-size: 100% 100%;
            pointer-events: none;
            z-index: 0;
            filter: blur(6px);
            animation: floatSpheres 28s ease-in-out infinite alternate;
            will-change: transform;
        }

        /* Subtle Dotted Matrix Grid Pattern */
        .stApp::after {
            content: "";
            position: fixed;
            top: -4%; left: -4%; width: 108vw; height: 108vh;
            background-image: radial-gradient(rgba(100, 116, 139, 0.22) 1.25px, transparent 1.25px);
            background-size: 26px 26px;
            pointer-events: none;
            z-index: 0;
            animation: floatSpheresGrid 34s ease-in-out infinite alternate;
            will-change: transform;
        }

        /* Sidebar Styling: Translucent Frosted Glass */
        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.72) !important;
            backdrop-filter: blur(28px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.85) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.03) !important;
            position: relative;
            z-index: 2;
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            gap: 0.75rem !important;
        }
        section[data-testid="stSidebar"] label {
            color: var(--text-primary) !important;
            font-size: 12px !important;
        }

        /* Continuous Pulse for Radar Beacon Dot */
        @keyframes radarBeacon {
            0% {
                transform: scale(0.9);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.75);
            }
            70% {
                transform: scale(1.15);
                box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
            }
            100% {
                transform: scale(0.9);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            }
        }

        /* Sidebar Brand Box (Creative Futuristic Glass Badge) */
        .sidebar-brand-box {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(238, 245, 255, 0.75) 100%) !important;
            backdrop-filter: blur(28px) saturate(190%) !important;
            -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
            border: 1.5px solid rgba(255, 255, 255, 0.95) !important;
            border-radius: 20px !important;
            padding: 16px 18px !important;
            margin-bottom: 16px !important;
            box-shadow: 0 15px 35px rgba(37, 99, 235, 0.08), inset 0 1px 2px rgba(255, 255, 255, 1.0) !important;
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
            cursor: default;
        }
        .sidebar-brand-box:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 20px 45px rgba(37, 99, 235, 0.15), inset 0 1px 2px rgba(255, 255, 255, 1.0) !important;
            border-color: rgba(37, 99, 235, 0.35) !important;
        }
        .sidebar-brand-title {
            font-family: 'Hosta', 'Host Grotesk', sans-serif !important;
            font-size: 20px !important;
            font-weight: 800 !important;
            color: #0f172a !important;
            letter-spacing: -0.5px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
        }
        .radar-beacon {
            display: inline-block;
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.85);
            animation: radarBeacon 1.8s infinite ease-in-out;
        }
        .sidebar-brand-sub {
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: 10px !important;
            font-weight: 800 !important;
            color: #475569 !important;
            letter-spacing: 1.1px !important;
            text-transform: uppercase !important;
            margin-top: 5px !important;
        }
        .sidebar-brand-microbadge {
            display: inline-block !important;
            margin-top: 8px !important;
            padding: 3px 8px !important;
            background: rgba(37, 99, 235, 0.08) !important;
            border: 1px solid rgba(37, 99, 235, 0.22) !important;
            border-radius: 7px !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: 9px !important;
            font-weight: 800 !important;
            letter-spacing: 0.6px !important;
            color: #2563eb !important;
        }
        .sidebar-brand-tag {
            display: none !important;
        }

        /* Sidebar Module Headers */
        .sidebar-section-header {
            white-space: nowrap !important;
            font-family: 'Hosta', 'Host Grotesk', sans-serif !important;
            font-weight: 700 !important;
            font-size: 11px !important;
            letter-spacing: 1px !important;
            color: #475569 !important;
            text-transform: uppercase !important;
            border-bottom: 1.5px solid rgba(226, 232, 240, 0.8) !important;
            padding-bottom: 6px !important;
            margin-top: 18px !important;
            margin-bottom: 10px !important;
            display: flex !important;
            align-items: center !important;
        }

        /* Elegant Minimalist Sliders: Charcoal Track & Solid White/Silver Glass Thumb */
        div[data-baseweb="slider"] {
            margin-top: 6px;
            margin-bottom: 6px;
        }
        div[data-baseweb="slider"] div[role="slider"] {
            background-color: #ffffff !important;
            border: 2px solid #0f172a !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15), 0 0 0 2px rgba(255, 255, 255, 0.8) !important;
            width: 18px !important;
            height: 18px !important;
            border-radius: 50% !important;
        }
        div[data-baseweb="slider"] > div > div:first-child {
            background: #cbd5e1 !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        div[data-baseweb="slider"] > div > div > div {
            background: #0f172a !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        div[data-testid="stThumbValue"] {
            color: #0f172a !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: 11px !important;
            font-weight: bold !important;
        }

        /* Form Controls & Selectboxes */
        div[data-baseweb="select"] > div {
            background-color: rgba(255, 255, 255, 0.85) !important;
            border: 1px solid rgba(0, 0, 0, 0.12) !important;
            color: #0f172a !important;
            border-radius: 12px !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-weight: bold !important;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.02) !important;
        }
        div[data-baseweb="select"] > div:hover {
            border-color: #0f172a !important;
        }
        div[data-baseweb="popover"] div[role="listbox"] {
            background-color: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(24px) !important;
            -webkit-backdrop-filter: blur(24px) !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            border-radius: 14px !important;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.08) !important;
            color: #0f172a !important;
        }

        /* Checkboxes & Radios */
        div[data-testid="stCheckbox"] label, div[data-testid="stRadio"] label {
            font-family: Arial, Helvetica, sans-serif !important;
            font-weight: bold !important;
            color: #0f172a !important;
        }
        div[data-testid="stCheckbox"] label span[role="checkbox"] {
            border-radius: 6px !important;
            border: 1.5px solid #94a3b8 !important;
            background: #ffffff !important;
        }
        div[data-testid="stCheckbox"] label span[role="checkbox"][aria-checked="true"] {
            background-color: #0f172a !important;
            border-color: #0f172a !important;
        }
        div[data-testid="stRadio"] label span[role="radio"][aria-checked="true"] {
            border-color: #0f172a !important;
        }
        div[data-testid="stRadio"] label span[role="radio"][aria-checked="true"]::after {
            background-color: #0f172a !important;
        }

        /* Universal Smooth Transitions for Interactive Elements */
        button,
        button *,
        div[data-testid="stMetric"],
        .soc-header-card,
        .sidebar-brand-box,
        div[data-testid="stImage"],
        div[data-testid="stExpander"],
        div[data-baseweb="slider"],
        div[data-baseweb="slider"] *,
        div[data-baseweb="select"] > div,
        input,
        select,
        textarea,
        div[data-testid="stCheckbox"],
        div[data-testid="stRadio"],
        div[data-testid="stFileUploader"] section {
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }

        /* Crisp Rounded Glass Action Buttons */
        button[kind="primary"] {
            background: #0f172a !important;
            border: 1px solid #0f172a !important;
            color: #ffffff !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-weight: bold !important;
            font-size: 13px !important;
            letter-spacing: 0.3px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.2) !important;
        }
        button[kind="primary"]:hover {
            background: #1e293b !important;
            border-color: #1e293b !important;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.25) !important;
            transform: translateY(-2px) !important;
        }
        button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.85) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(0, 0, 0, 0.12) !important;
            color: #0f172a !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-weight: bold !important;
            font-size: 13px !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04), inset 0 1px 1px rgba(255, 255, 255, 0.95) !important;
        }
        button[kind="secondary"]:hover {
            background: #ffffff !important;
            border-color: #0f172a !important;
            color: #0f172a !important;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18) !important;
            transform: translateY(-2px) !important;
        }

        /* Telemetry Header Card (Light Glassmorphism Bar) */
        .soc-header-card {
            background: var(--glass-card-bg);
            backdrop-filter: blur(28px) saturate(190%);
            -webkit-backdrop-filter: blur(28px) saturate(190%);
            border: 1.5px solid var(--glass-card-border);
            border-radius: 20px;
            padding: 18px 24px;
            margin-bottom: 20px;
            box-shadow: var(--glass-card-shadow);
            position: relative;
        }
        .soc-header-card:hover {
            box-shadow: 0 25px 50px rgba(37, 99, 235, 0.08), inset 0 1px 2px rgba(255, 255, 255, 1) !important;
            border-color: rgba(37, 99, 235, 0.25) !important;
        }
        .soc-classification-strip {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(0, 0, 0, 0.06);
            padding-bottom: 8px;
            margin-bottom: 12px;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 0.8px;
            color: var(--text-muted);
        }
        .soc-class-tag {
            color: var(--text-primary);
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .soc-class-meta {
            color: var(--text-muted);
            font-weight: bold;
        }
        .soc-title-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 12px;
        }
        .soc-main-heading {
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: var(--text-primary);
        }
        .soc-main-heading span.accent {
            color: var(--text-primary);
            opacity: 0.85;
        }
        .soc-main-heading span.dim {
            color: var(--text-muted);
            font-family: Arial, Helvetica, sans-serif !important;
            font-weight: bold;
            font-size: 15px;
            margin-left: 8px;
        }
        .soc-telemetry-hud-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 11px;
            font-weight: bold;
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.55);
            padding: 8px 14px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.7);
        }
        .soc-telem-pill {
            display: flex;
            gap: 6px;
        }
        .soc-telem-pill .k {
            color: var(--text-muted);
            font-weight: bold;
        }
        .soc-telem-pill .v {
            color: var(--text-primary);
            font-weight: bold;
        }
        .pulsing-dot-green {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
            animation: pulse-green 1.8s infinite;
        }
        @keyframes pulse-green {
            0% { opacity: 0.4; }
            50% { opacity: 1; }
            100% { opacity: 0.4; }
        }

        /* Metric HUD Cards (Clean Frosted Glass Tiles with High Contrast & Hover) */
        div[data-testid="stMetric"] {
            background: var(--glass-card-bg) !important;
            backdrop-filter: blur(24px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
            border: 1.5px solid var(--glass-card-border) !important;
            border-radius: 20px !important;
            padding: 16px 20px !important;
            box-shadow: var(--glass-card-shadow) !important;
            position: relative;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-4px) scale(1.01) !important;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.08) !important;
            border-color: rgba(37, 99, 235, 0.4) !important;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricLabel"] span,
        div[data-testid="stMetricLabel"] div,
        div[data-testid="stMetricLabel"] label {
            color: #334155 !important;
            font-family: Arial, sans-serif !important;
            font-weight: 800 !important;
            font-size: 11px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
        }
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] *,
        div[data-testid="stMetricValue"] div {
            color: #0f172a !important;
            font-family: 'Hosta', 'Host Grotesk', sans-serif !important;
            font-weight: 800 !important;
            font-size: 26px !important;
            letter-spacing: -0.5px !important;
        }
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] *,
        div[data-testid="stMetricDelta"] svg {
            font-family: Arial, sans-serif !important;
            font-weight: 800 !important;
            font-size: 11px !important;
            color: #475569 !important;
        }

        /* Viewport HUD Header Bar */
        .viewport-hud-topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 16px;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 11px;
            font-weight: bold;
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.82);
            backdrop-filter: blur(24px) saturate(180%);
            -webkit-backdrop-filter: blur(24px) saturate(180%);
            border: 1px solid var(--glass-card-border);
            border-bottom: none;
            border-radius: 20px 20px 0 0;
            margin-top: 14px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        .viewport-hud-title {
            font-family: 'Hosta', 'Host Grotesk', sans-serif !important;
            font-weight: 700 !important;
            color: var(--text-primary);
            letter-spacing: 0.4px;
        }
        .viewport-hud-badge {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.25);
            color: #059669;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: bold;
        }

        /* Frosted Glass Single-Layer Alert Container (No Nested Or Overlapping Boxes) */
        div[data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(28px) saturate(190%) !important;
            -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
            border: 1px solid rgba(226, 232, 240, 0.9) !important;
            border-left: 6px solid #2563eb !important;
            border-radius: 16px !important;
            color: #0f172a !important;
            font-family: Arial, sans-serif !important;
            font-weight: bold !important;
            font-size: 13px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04), inset 0 1px 1px rgba(255, 255, 255, 1.0) !important;
            padding: 14px 20px !important;
            margin-bottom: 16px !important;
            overflow: hidden !important;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }

        /* Completely reset all inner nested wrappers so NO inner boxes or secondary borders appear */
        div[data-testid="stAlert"] > div,
        div[data-testid="stAlert"] div,
        div[data-testid="stAlert"] section,
        div[data-testid="stAlertContainer"],
        div[data-testid^="stAlertContent"],
        div[data-testid="stAlert"] [data-testid="stAlertContainer"],
        div[data-testid="stAlert"] [data-testid^="stAlertContent"] {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            border-left: none !important;
            border-right: none !important;
            border-top: none !important;
            border-bottom: none !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Frosted Glass Warning Alert Banner (Surveillance Paused Notice - Single Clean Box) */
        div[data-testid="stAlert"]:has([data-testid*="Warning"]),
        div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
            background: rgba(255, 255, 255, 0.96) !important;
            border: 1px solid rgba(245, 158, 11, 0.4) !important;
            border-left: 6px solid #f59e0b !important;
            box-shadow: 0 12px 30px rgba(245, 158, 11, 0.08), inset 0 1px 1px rgba(255, 255, 255, 1.0) !important;
        }
        div[data-testid="stAlert"]:has([data-testid*="Warning"]) *,
        div[data-testid="stAlert"]:has([data-testid*="Warning"]) p,
        div[data-testid="stAlert"]:has([data-testid*="Warning"]) span {
            color: #0f172a !important;
            font-family: Arial, sans-serif !important;
            font-weight: 800 !important;
            font-size: 13px !important;
        }
        div[data-testid="stAlert"]:has([data-testid*="Warning"]) svg {
            fill: #f59e0b !important;
            color: #f59e0b !important;
        }

        /* Critical Perimeter Threat Error Alert Banner (Pulsing Red Single Box) */
        div[data-testid="stAlert"]:has([data-testid*="Error"]),
        div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
            background: rgba(254, 242, 242, 0.95) !important;
            backdrop-filter: blur(24px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
            border: 1px solid #fca5a5 !important;
            border-left: 6px solid #ef4444 !important;
            border-radius: 16px !important;
            color: #991b1b !important;
            box-shadow: 0 12px 30px rgba(239, 68, 68, 0.16) !important;
            animation: pulse-red-alert 1.5s infinite ease-in-out !important;
        }
        div[data-testid="stAlert"]:has([data-testid*="Error"]) *,
        div[data-testid="stAlert"]:has([data-testid*="Error"]) p,
        div[data-testid="stAlert"]:has([data-testid*="Error"]) span {
            color: #991b1b !important;
            font-family: Arial, sans-serif !important;
            font-weight: 800 !important;
            font-size: 13px !important;
        }
        @keyframes pulse-red-alert {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.35); }
            50% { box-shadow: 0 0 20px 2px rgba(239, 68, 68, 0.2); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.35); }
        }

        /* Snapshot and Image Framing */
        div[data-testid="stImage"] {
            background: rgba(255, 255, 255, 0.72) !important;
            backdrop-filter: blur(24px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
            border: 1px solid var(--glass-card-border) !important;
            border-radius: 18px !important;
            padding: 8px !important;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.03), inset 0 1px 1px rgba(255, 255, 255, 0.95) !important;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        div[data-testid="stImage"] img {
            border-radius: 12px !important;
        }
        div[data-testid="stImage"]:hover {
            border-color: rgba(37, 99, 235, 0.35) !important;
            box-shadow: 0 18px 40px rgba(37, 99, 235, 0.09) !important;
            transform: translateY(-2px);
        }
        div[data-testid="stImageCaption"] {
            color: #334155 !important;
            font-family: Arial, Helvetica, sans-serif !important;
            font-weight: bold !important;
            font-size: 11px !important;
            line-height: 1.4 !important;
            margin-top: 6px !important;
            background: rgba(255, 255, 255, 0.8) !important;
            padding: 6px 10px !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.9) !important;
        }

        /* Dataframe & Tables */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--glass-card-border) !important;
            border-radius: 18px !important;
            background-color: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(24px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
            box-shadow: var(--glass-card-shadow) !important;
            overflow: hidden !important;
        }

        /* File Uploader (Frosted Glass Dropzone & Clean Secondary Action) */
        div[data-testid="stFileUploader"] section {
            border: 1.5px dashed rgba(37, 99, 235, 0.35) !important;
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-radius: 16px !important;
            padding: 16px 14px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
            text-align: center !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02) !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stFileUploader"] section:hover {
            border-color: #2563eb !important;
            background: rgba(255, 255, 255, 0.95) !important;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.08) !important;
        }
        div[data-testid="stFileUploader"] section > div {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
            width: 100% !important;
        }

        /* Suppress duplicate text ligature icons in the file uploader button */
        div[data-testid="stFileUploader"] section button [data-testid="stIconMaterial"],
        div[data-testid="stFileUploader"] section button [data-testid*="Icon"],
        div[data-testid="stFileUploader"] section button [aria-hidden="true"],
        div[data-testid="stFileUploader"] section button > span:first-child:not(:only-child) {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            font-size: 0 !important;
            line-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            position: absolute !important;
            pointer-events: none !important;
        }

        div[data-testid="stFileUploader"] button,
        div[data-testid="stFileUploader"] button[kind="secondary"],
        div[data-testid="stFileUploader"] section button {
            background: #0f172a !important;
            color: #ffffff !important;
            border: 1px solid #0f172a !important;
            border-radius: 12px !important;
            font-family: Arial, sans-serif !important;
            font-weight: 800 !important;
            font-size: 13px !important;
            letter-spacing: 0.5px !important;
            padding: 8px 24px !important;
            min-width: 120px !important;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15) !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
            display: inline-flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        div[data-testid="stFileUploader"] button:hover,
        div[data-testid="stFileUploader"] button[kind="secondary"]:hover,
        div[data-testid="stFileUploader"] section button:hover {
            background: #1e293b !important;
            border-color: #1e293b !important;
            color: #ffffff !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.22) !important;
        }
        div[data-testid="stFileUploader"] section button > span:last-child {
            display: inline-block !important;
            color: #ffffff !important;
            font-family: Arial, sans-serif !important;
            font-weight: 800 !important;
            font-size: 13px !important;
            white-space: nowrap !important;
        }
        div[data-testid="stFileUploader"] button *,
        div[data-testid="stFileUploader"] button[kind="secondary"] *,
        div[data-testid="stFileUploader"] section button * {
            color: #ffffff !important;
            fill: #ffffff !important;
            white-space: nowrap !important;
        }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            margin-top: 6px !important;
            width: 100% !important;
        }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span,
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] small {
            display: block !important;
            font-family: Arial, sans-serif !important;
            font-size: 11px !important;
            color: #475569 !important;
            font-weight: 700 !important;
            text-align: center !important;
            white-space: normal !important;
            word-break: break-word !important;
            overflow: visible !important;
            line-height: 1.4 !important;
        }
        div[data-testid="stExpander"] {
            background: var(--glass-card-bg) !important;
            backdrop-filter: blur(20px) !important;
            border: 1px solid var(--glass-card-border) !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.02) !important;
        }

        /* Section Headings */
        .section-title {
            font-family: 'Hosta', 'Host Grotesk', sans-serif !important;
            font-size: 14px !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
            letter-spacing: 0.4px !important;
            margin-top: 20px !important;
            margin-bottom: 10px !important;
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
        }

        /* Sidebar Telemetry Box */
        .sidebar-telemetry-box {
            font-family: Arial, Helvetica, sans-serif;
            font-weight: bold;
            font-size: 11px;
            color: var(--text-secondary);
            line-height: 1.7;
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            padding: 10px 14px;
            border-radius: 14px;
            border: 1px solid var(--glass-card-border);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02);
        }
        .sidebar-telemetry-box .telemetry-bullet {
            color: var(--text-primary);
            margin-right: 6px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar: Defense Console Layout
    st.sidebar.markdown("""
        <div class="sidebar-brand-box">
            <div class="sidebar-brand-title">
                <span>TRIP-WIRE</span>
                <span class="radar-beacon"></span>
            </div>
            <div class="sidebar-brand-sub">SPATIAL DEFENSE // C2 INTELLIGENCE</div>
            <div class="sidebar-brand-microbadge">[REAL-TIME BEV • 4D KALMAN]</div>
        </div>
    """, unsafe_allow_html=True)

    # Section 1: Perception & AI Engine
    st.sidebar.markdown('<div class="sidebar-section-header">PERCEPTION & AI ENGINE</div>', unsafe_allow_html=True)

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

    conf_thresh = st.sidebar.slider("Detection Confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    infer_imgsz = st.sidebar.select_slider(
        "Inference Resolution (imgsz)",
        options=[384, 480, 640],
        value=480,
        help="Lower resolutions reduce compute cost; validate accuracy for your camera."
    )

    # Section 2: Spatial Geometry & Calibration
    st.sidebar.markdown('<div class="sidebar-section-header">SPATIAL GEOMETRY & CALIBRATION</div>', unsafe_allow_html=True)

    feed_option = st.sidebar.selectbox(
        "Select Video Feed",
        [
            "Flagship: Multi-Threat Perimeter Breach (Calibrated Benchmark)",
            "Template: Corridor Tripwire Crossing (Breach Demo)",
            "Template: High-Speed Fleeing Intruder (Alarm Demo)",
            "Walking Corridor CCTV (Full 60s)",
            "Hallway Intrusion",
            "Pedestrian Crossing",
            "Live Laptop Camera (Webcam 0)",
            "External Camera (Webcam 1)",
            "Browser Web Camera (Any Laptop / Cloud / Remote)",
            "Upload Custom MP4"
        ],
        index=0
    )

    video_path = None
    if feed_option == "Flagship: Multi-Threat Perimeter Breach (Calibrated Benchmark)":
        video_path = "data/videos/flagship_multi_threat_benchmark.mp4"
    elif feed_option == "Template: Corridor Tripwire Crossing (Breach Demo)":
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
    elif feed_option == "Browser Web Camera (Any Laptop / Cloud / Remote)":
        video_path = "browser_camera"
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

    zone_preset = st.sidebar.radio(
        "Security Zone Layout",
        ["Realistic Security (Side Vault & Checkpoint)", "Demo Stress Test (Center Hallway)"],
        index=0,
        help="Realistic layout monitors off-limits doors/alcoves. Stress test places the zone in the center walkway."
    )
    calibration_upload = st.sidebar.file_uploader('Calibration JSON for this feed', type=['json'])
    use_demo_calib = st.sidebar.checkbox(
        "Apply Floor Calibration (Enable Velocity & Radar)",
        value=True,
        help="Applies default calibrated floor homography to estimate ground coordinates, metric velocity (km/h), and BEV radar blips when camera-specific calibration is not loaded."
    )
    show_grid = st.sidebar.checkbox("Project Perspective Ground Grid", value=True)

    # Section 3: Thresholds & Threat Dispatch
    st.sidebar.markdown('<div class="sidebar-section-header">THRESHOLDS & THREAT DISPATCH</div>', unsafe_allow_html=True)
    speed_alarm = st.sidebar.slider("Speed Alarm Threshold (km/h)", min_value=1.0, max_value=25.0, value=5.0, step=0.5)
    dwell_confirm_sec = st.sidebar.slider(
        "Breach Dwell / Debounce (sec)",
        min_value=0.0,
        max_value=4.0,
        value=1.5,
        step=0.5,
        help="Target must linger inside zone for this duration to confirm breach. Prevents false alarms during normal transit."
    )
    # 🔊 Audible Alarm Siren (test harness compatibility hook)
    enable_audio_alarm = st.sidebar.checkbox(
        "Audible Alarm Siren",
        value=True,
        help="Plays an audible security siren when an active perimeter breach or intrusion threat is detected."
    )
    if "siren_controller" in st.session_state:
        st.session_state.siren_controller.set_enabled(enable_audio_alarm)

    # Section 4: Mission Dispatch Controls
    st.sidebar.markdown('<div class="sidebar-section-header">MISSION DISPATCH CONTROLS</div>', unsafe_allow_html=True)
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("Start Live", type="primary", key="btn_start_live", **WIDTH_KWARGS):
            st.session_state.is_running = True
            st.rerun()
    with col_btn2:
        if st.button("Stop Feed", key="btn_stop_feed", **WIDTH_KWARGS):
            st.session_state.is_running = False
            if st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                except Exception:
                    pass
                st.session_state.cap = None
            st.session_state.current_feed = None
            st.rerun()

    # Section 5: Forensic Export
    if st.session_state.incident_log:
        st.sidebar.markdown('<div class="sidebar-section-header">FORENSIC EXPORT</div>', unsafe_allow_html=True)
        with st.sidebar.expander(f"Export Incident Logs ({len(st.session_state.incident_log)})", expanded=True):
            df_sb = pd.DataFrame(st.session_state.incident_log[::-1])
            csv_sb = df_sb.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_sb, "tripwire_incident_audit.csv", "text/csv", key="sb_download_csv", **WIDTH_KWARGS)
            json_sb = json.dumps(st.session_state.incident_log, indent=2).encode('utf-8')
            st.download_button("Download JSON", json_sb, "tripwire_forensic_report.json", "application/json", key="sb_download_json", **WIDTH_KWARGS)

    # Section 6: Telemetry Architecture
    st.sidebar.markdown('<div class="sidebar-section-header">TELEMETRY ARCHITECTURE</div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
        <div class="sidebar-telemetry-box">
            <div><span class="telemetry-bullet">•</span> ByteTrack Two-Stage Bipartite</div>
            <div><span class="telemetry-bullet">•</span> 4D Kalman (Joseph Form)</div>
            <div><span class="telemetry-bullet">•</span> 17-Joint Kinetic Classifier</div>
            <div><span class="telemetry-bullet">•</span> Hartley-Normalized DLT</div>
            <div><span class="telemetry-bullet">•</span> 20m x 20m BEV Radar Map</div>
        </div>
    """, unsafe_allow_html=True)

    # Main Area: Telemetry Header
    st.markdown("""
        <div class="soc-header-card">
            <div class="soc-classification-strip">
                <div class="soc-class-tag">
                    <span class="pulsing-dot-green"></span> RESTRICTED // SPATIAL DEFENSE INTELLIGENCE &amp; THREAT OPERATIONS
                </div>
                <div class="soc-class-meta">
                    POSTURE: SECURE &bull; STATION: ALPHA &bull; REV: v2.4
                </div>
            </div>
            <div class="soc-title-row">
                <div class="soc-main-heading">
                    TRIP-WIRE <span class="accent">// SPATIAL DEFENSE</span> <span class="dim">&amp; BEV RADAR HUD</span>
                </div>
            </div>
            <div class="soc-telemetry-hud-strip">
                <div class="soc-telem-pill"><span class="k">SURVEY:</span> <span class="v">DLT HOMOGRAPHY [CALIBRATED]</span></div>
                <div class="soc-telem-pill"><span class="k">KINEMATICS:</span> <span class="v">4D KALMAN FILTER [CV-STABILIZED]</span></div>
                <div class="soc-telem-pill"><span class="k">AI ENGINE:</span> <span class="v">17-PT COCO KINETICS</span></div>
                <div class="soc-telem-pill"><span class="k">PROJECTION:</span> <span class="v">20m x 20m BEV GROUND PLANE</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Dynamic Top Metric HUD Cards (5 panels)
    col1, col2, col3, col4, col5 = st.columns(5)
    metric_status = col1.empty()
    metric_fps = col2.empty()
    metric_tracks = col3.empty()
    metric_threat = col4.empty()
    metric_breaches = col5.empty()

    is_active = st.session_state.is_running
    metric_status.metric("Stream Status", "Active Feed" if is_active else "Standby", "Online (Live)" if is_active else "Awaiting Engage")
    metric_fps.metric("Processing Speed", "Standby", "0.0 FPS")
    metric_tracks.metric("Active Targets", "0", "Awaiting Feed")
    metric_threat.metric("Threat Index", "Nominal", "DEFCON 4 // Secure")
    metric_breaches.metric("Intrusion Breaches", "0", "Zero Violations")

    # Main Display Area
    alert_banner_placeholder = st.empty()
    audio_siren_placeholder = st.empty()
    viewport_header_placeholder = st.empty()
    dual_view_placeholder = st.empty()
    status_text = st.empty()

    # Evidence Locker and Incident Table Placeholders
    evidence_placeholder = st.empty()
    table_placeholder = st.empty()

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

    if video_path == "browser_camera":
        run_browser_camera_pipeline(
            feed_option, zone_preset, use_pose, use_demo_calib, imgsz, conf_threshold,
            speed_alarm, dwell_confirm_sec, metric_status, metric_fps, metric_tracks,
            metric_threat, metric_breaches, alert_banner_placeholder, dual_view_placeholder,
            viewport_header_placeholder, key="browser_cam_client"
        )

    elif st.session_state.is_running and video_path is not None:
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            cap, actual_src, backend_name, err = open_capture_source(video_path)
            if cap is None or not cap.isOpened():
                if isinstance(video_path, int):
                    status_text.warning(
                        f"Hardware camera index #{video_path} is not attached to this server host (e.g. running on Streamlit Cloud or remote container). "
                        "Switched automatically to live Client Browser Camera below — allow camera access in your browser to begin live surveillance!"
                    )
                    run_browser_camera_pipeline(
                        feed_option, zone_preset, use_pose, use_demo_calib, imgsz, conf_threshold,
                        speed_alarm, dwell_confirm_sec, metric_status, metric_fps, metric_tracks,
                        metric_threat, metric_breaches, alert_banner_placeholder, dual_view_placeholder,
                        viewport_header_placeholder, key="fallback_browser_cam"
                    )
                    return
                else:
                    metric_status.metric("Stream Status", "Standby", "Awaiting Engage")
                    if err:
                        status_text.error(err)
                    else:
                        status_text.error(f"Could not open stream for '{feed_option}'.")
                    st.session_state.is_running = False
                    return

            if isinstance(video_path, int) and actual_src != video_path:
                status_text.info(f"Target webcam index #{video_path} was unavailable; automatically bound to active camera #{actual_src} ({backend_name}).")
                video_path = actual_src

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
            st.warning('This feed is uncalibrated. Ground positions, metric speed, and spatial alarms are disabled. Check "Apply Floor Calibration" or upload a calibration JSON for this camera.')
        calibration_key = (str(video_path), transformer.H.tobytes() if transformer.is_calibrated else None, zone_preset, use_pose, use_demo_calib)
        if st.session_state.get('_calibration_key') != calibration_key:
            st.session_state.tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
            st.session_state.zone_monitor = ZoneMonitor()
            st.session_state._calibration_key = calibration_key
        tracker = st.session_state.tracker

        zones = get_default_zones(feed_option, preset=zone_preset)

        frame_count = 0
        t0 = time.time()
        last_siren_time = 0.0
        consecutive_read_failures = 0
        is_webcam = isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit())
        MAX_READ_FAILURES = 10 if is_webcam else 1

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
                        consecutive_read_failures += 1
                        if consecutive_read_failures < MAX_READ_FAILURES:
                            time.sleep(0.04)
                            continue
                        status_text.error(f"Video stream or camera feed '{feed_option}' was disconnected or ended.")
                        st.session_state.is_running = False
                        if st.session_state.cap is not None:
                            try:
                                st.session_state.cap.release()
                            except Exception:
                                pass
                            st.session_state.cap = None
                        break
                else:
                    consecutive_read_failures = 0

                frame_elapsed_dt = clock.tick()
                frame_count += 1
                t_frame_start = time.time()

                # Clean frame for inference
                display_frame = frame.copy()

                # Grid & Security Zone Projection on Camera View
                if transformer.is_calibrated:
                    if show_grid:
                        max_y = 12.0 if ("Pedestrian" in feed_option or "Flagship" in feed_option) else (4.0 if ("Laptop" in feed_option or "Webcam" in feed_option) else 6.0)
                        segments = transformer.project_ground_grid_to_image(display_frame.shape[:2], grid_step_m=2.0, max_visible_y=max_y)
                        grid_overlay = display_frame.copy()
                        for pt1, pt2 in segments:
                            cv2.line(grid_overlay, pt1, pt2, (60, 80, 90), 1, cv2.LINE_AA)
                        cv2.addWeighted(grid_overlay, 0.4, display_frame, 0.6, 0, display_frame)

                    poly_overlay = display_frame.copy()
                    has_polys = False
                    for z in zones:
                        if z.zone_type == "polygon":
                            try:
                                pts = [transformer.world_to_image(x, y) for x, y in z.world_coords]
                                pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(poly_overlay, [pts_np], (0, 140, 255))
                                has_polys = True
                            except Exception:
                                pass

                    if has_polys:
                        cv2.addWeighted(poly_overlay, 0.30, display_frame, 0.70, 0, display_frame)

                    for z in zones:
                        if z.zone_type == "polygon":
                            try:
                                pts = [transformer.world_to_image(x, y) for x, y in z.world_coords]
                                pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.polylines(display_frame, [pts_np], True, (0, 185, 255), 2, cv2.LINE_AA)
                                for pt in pts:
                                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 4, (0, 220, 255), -1, cv2.LINE_AA)
                                lbl_x = int(min(p[0] for p in pts)) + 8
                                lbl_y = int(min(p[1] for p in pts)) + 20
                                (lw, lh), _ = cv2.getTextSize(z.name, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                                cv2.rectangle(display_frame, (lbl_x - 3, lbl_y - lh - 4), (lbl_x + lw + 4, lbl_y + 4), (15, 20, 25), -1)
                                cv2.rectangle(display_frame, (lbl_x - 3, lbl_y - lh - 4), (lbl_x + lw + 4, lbl_y + 4), (0, 165, 255), 1)
                                cv2.putText(display_frame, z.name, (lbl_x, lbl_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
                            except Exception:
                                pass
                        elif z.zone_type == "tripwire":
                            try:
                                p1 = transformer.world_to_image(*z.world_coords[0])
                                p2 = transformer.world_to_image(*z.world_coords[1])
                                ip1 = (int(p1[0]), int(p1[1]))
                                ip2 = (int(p2[0]), int(p2[1]))
                                wire_overlay = display_frame.copy()
                                cv2.line(wire_overlay, ip1, ip2, (0, 0, 255), 4, cv2.LINE_AA)
                                cv2.addWeighted(wire_overlay, 0.4, display_frame, 0.6, 0, display_frame)
                                cv2.line(display_frame, ip1, ip2, (50, 50, 255), 2, cv2.LINE_AA)
                                cv2.circle(display_frame, ip1, 5, (0, 0, 255), -1, cv2.LINE_AA)
                                cv2.circle(display_frame, ip2, 5, (0, 0, 255), -1, cv2.LINE_AA)
                                mid_x = (ip1[0] + ip2[0]) // 2
                                mid_y = (ip1[1] + ip2[1]) // 2 - 8
                                (lw, lh), _ = cv2.getTextSize(z.name, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                                bx = max(10, mid_x - lw // 2)
                                cv2.rectangle(display_frame, (bx - 3, mid_y - lh - 4), (bx + lw + 4, mid_y + 4), (15, 20, 25), -1)
                                cv2.rectangle(display_frame, (bx - 3, mid_y - lh - 4), (bx + lw + 4, mid_y + 4), (0, 0, 255), 1)
                                cv2.putText(display_frame, z.name, (bx, mid_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 150, 255), 1, cv2.LINE_AA)
                            except Exception:
                                pass

                # Deep Visual Perception — always infer on the ORIGINAL clean frame
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
                    zones=zones,
                    elapsed_dt=frame_elapsed_dt
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
                        tag_prefix = "INTRUSION BREACH"
                    elif is_monitoring:
                        color = (0, 215, 255)    # Amber: Monitoring / Grace Period
                        tag_prefix = "MONITORING"
                    else:
                        color = (0, 255, 0)      # Green: Authorized Target
                        tag_prefix = "SECURE"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                    speed_label = f"{speed_kmh:.1f}km/h" if w_pos is not None else "speed unavailable"
                    tag_label = f"{tag_prefix} #{trk['track_id']} | {posture_label.upper()} | {speed_label}"
                    if breach_reasons:
                        tag_label += f" | {breach_reasons[0]}"
                    (tw, th), baseline = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                    tag_y = max(th + 6, y1 - 8)
                    cv2.rectangle(display_frame, (x1, tag_y - th - 4), (x1 + tw + 6, tag_y + baseline), (15, 20, 25), -1)
                    cv2.rectangle(display_frame, (x1, tag_y - th - 4), (x1 + tw + 6, tag_y + baseline), color, 1)
                    cv2.putText(display_frame, tag_label, (x1 + 3, tag_y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

                    new_alarm = next((e for e in frame_events if e['track_id'] == tid and
                                      e['event'] in ('LOITER', 'CROSS_IN', 'CROSS_OUT', 'SPEED_START', 'ACTION_START')), None)
                    if new_alarm is not None:
                        ih, iw = frame.shape[:2]
                        pad_w = int((x2 - x1) * 0.1)
                        pad_h = int((y2 - y1) * 0.1)
                        cx1, cy1, cx2, cy2 = max(0, x1 - pad_w), max(0, y1 - pad_h), min(iw, x2 + pad_w), min(ih, y2 + pad_h)
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
                dual_view = BEVRadarRenderer.create_dual_view(display_frame, radar_img)

                dual_view_resized = cv2.resize(dual_view, (1280, 480), interpolation=cv2.INTER_AREA)
                dual_view_rgb = cv2.cvtColor(dual_view_resized, cv2.COLOR_BGR2RGB)
                
                viewport_header_placeholder.markdown("""
                    <div class="viewport-hud-topbar">
                        <div class="viewport-hud-title">
                            <span>PRIMARY OPTICAL FOV (LEFT) // BIRD'S EYE VIEW RADAR 20m (RIGHT)</span>
                        </div>
                        <div class="viewport-hud-badge">LIVE SYNCHRONIZED</div>
                    </div>
                """, unsafe_allow_html=True)
                dual_view_placeholder.image(dual_view_rgb, channels="RGB", **IMAGE_KWARGS)

                elapsed = time.time() - t0
                fps = frame_count / max(0.001, elapsed)

                # Dynamically update live metrics
                threat_label = "CRITICAL" if breach_count > 0 else ("ELEVATED" if any(t.get('threat_level', 1) > 1 for t in active_tracks) else "NOMINAL")
                threat_delta = "BREACH ACTIVE" if breach_count > 0 else ("Warning: Elevated Kinetics" if any(t.get('threat_level', 1) > 1 for t in active_tracks) else "Spatial Grid Secure")
                feed_short = (feed_option[:14] + "…") if len(feed_option) > 15 else feed_option

                metric_status.metric("Stream Status", "Active Feed", f"{feed_short}")
                metric_fps.metric("Processing Speed", f"{fps:.1f} FPS", f"Infer: {infer_ms:.1f}ms | Track: {track_ms:.1f}ms")
                metric_tracks.metric("Active Targets", f"{len(active_tracks)}", f"{len(active_tracks)} in camera frame")
                metric_threat.metric("Threat Index", threat_label, threat_delta)
                metric_breaches.metric("Intrusion Breaches", f"{breach_count}", "ALARM ACTIVE" if breach_count > 0 else "Secure")

                # Visual Threat Banner & Native Web Audio Siren Synthesizer
                if breach_count > 0:
                    alert_banner_placeholder.error("CRITICAL PERIMETER BREACH ACTIVE — INTRUDER DETECTED IN RESTRICTED ZONE")
                    if "siren_controller" in st.session_state:
                        st.session_state.siren_controller.sync_with_events(events=frame_events, states=alarm_states, breach_count=breach_count)
                    if enable_audio_alarm and components is not None:
                        if now_ts - last_siren_time >= 0.85:
                            last_siren_time = now_ts
                            alarm_token = int(now_ts * 1000)
                            siren_html = f"""
                            <div id="tripwire-siren-{alarm_token}" style="display:none;"></div>
                            <script>
                            (function() {{
                                try {{
                                    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                                    if (!AudioContextClass) return;
                                    const ctx = new AudioContextClass();
                                    if (ctx.state === 'suspended') {{
                                        ctx.resume();
                                    }}
                                    const t = ctx.currentTime;
                                    const osc = ctx.createOscillator();
                                    const gain = ctx.createGain();

                                    // Crisp two-tone tactical security alarm siren pulse (880Hz / 659Hz)
                                    osc.type = 'sawtooth';
                                    osc.frequency.setValueAtTime(880, t);
                                    osc.frequency.setValueAtTime(659, t + 0.18);
                                    osc.frequency.setValueAtTime(880, t + 0.36);
                                    osc.frequency.setValueAtTime(659, t + 0.54);

                                    gain.gain.setValueAtTime(0.001, t);
                                    gain.gain.exponentialRampToValueAtTime(0.25, t + 0.02);
                                    gain.gain.setValueAtTime(0.25, t + 0.65);
                                    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.72);

                                    osc.connect(gain);
                                    gain.connect(ctx.destination);

                                    osc.start(t);
                                    osc.stop(t + 0.73);
                                    osc.onended = function() {{
                                        try {{ ctx.close(); }} catch(e) {{}}
                                    }};
                                }} catch (err) {{
                                    console.warn("Web Audio alarm synthesizer error:", err);
                                }}
                            }})();
                            </script>
                            """
                            with audio_siren_placeholder:
                                components.html(siren_html, height=0, width=0)
                else:
                    if "siren_controller" in st.session_state:
                        st.session_state.siren_controller.sync_with_events(breach_count=0)
                    alert_banner_placeholder.empty()
                    audio_siren_placeholder.empty()

                status_text.info(f"Pipeline FPS: **{fps:.1f}** | Inference: **{infer_ms:.1f}ms** | Tracking: **{track_ms:.1f}ms** | Active Targets: **{len(active_tracks)}** | Breaches: **{breach_count}** | Feed: **{feed_option}**")

                # Render Incident Audit Log and Evidence Locker (Live updates)
                if st.session_state.evidence_locker:
                    with evidence_placeholder.container():
                        st.markdown("""
                            <div class="section-title">
                                <span>FORENSIC EVIDENCE LOCKER // INTERCEPT CAPTURES</span>
                            </div>
                        """, unsafe_allow_html=True)
                        cols = st.columns(min(4, len(st.session_state.evidence_locker)))
                        for idx_snap, snap in enumerate(reversed(st.session_state.evidence_locker[-4:])):
                            with cols[idx_snap % 4]:
                                st.image(snap["image"], caption=f"TARGET #{snap['track_id']} | {snap['time_str']} | {snap['action']}\n{snap['reason']} ({snap['speed']})", **IMAGE_KWARGS)

                if st.session_state.incident_log:
                    with table_placeholder.container():
                        st.markdown(f"""
                            <div class="section-title">
                                <span>INCIDENT AUDIT TRAIL // REAL-TIME EVENT STREAM ({len(st.session_state.incident_log)} EVENTS LOGGED)</span>
                            </div>
                        """, unsafe_allow_html=True)
                        df_log = pd.DataFrame(st.session_state.incident_log[::-1])
                        st.dataframe(df_log, height=180, **DATAFRAME_KWARGS)
                        st.caption("*To export CSV or JSON forensics reports, click **Stop Feed** in the sidebar or expand **Export Incident Logs**.*")

                time.sleep(0.015)

        finally:
            alert_banner_placeholder.empty()
            audio_siren_placeholder.empty()
            cap.release()
            st.session_state.cap = None

    elif not st.session_state.is_running:
        alert_banner_placeholder.empty()
        audio_siren_placeholder.empty()
        status_text.warning("Surveillance pipeline is paused. Click **Start Live** in the sidebar to run.")
        if st.session_state.evidence_locker:
            with evidence_placeholder.container():
                st.markdown("""
                    <div class="section-title">
                        <span>FORENSIC EVIDENCE LOCKER // INTERCEPT CAPTURES</span>
                    </div>
                """, unsafe_allow_html=True)
                cols = st.columns(min(4, len(st.session_state.evidence_locker)))
                for idx_snap, snap in enumerate(reversed(st.session_state.evidence_locker[-4:])):
                    with cols[idx_snap % 4]:
                        st.image(snap["image"], caption=f"TARGET #{snap['track_id']} | {snap['time_str']} | {snap['action']}\n{snap['reason']} ({snap['speed']})", **IMAGE_KWARGS)

        if st.session_state.incident_log:
            with table_placeholder.container():
                st.markdown(f"""
                    <div class="section-title">
                        <span>INCIDENT AUDIT TRAIL // REAL-TIME EVENT STREAM ({len(st.session_state.incident_log)} TOTAL EVENTS)</span>
                    </div>
                """, unsafe_allow_html=True)
                df_log = pd.DataFrame(st.session_state.incident_log[::-1])
                st.dataframe(df_log, height=180, **DATAFRAME_KWARGS)
                col_csv, col_json = st.columns(2)
                with col_csv:
                    csv_data = df_log.to_csv(index=False).encode('utf-8')
                    st.download_button("Export Audit Log (CSV)", csv_data, "tripwire_incident_audit.csv", "text/csv", key="paused_export_csv", **WIDTH_KWARGS)
                with col_json:
                    json_data = json.dumps(st.session_state.incident_log, indent=2).encode('utf-8')
                    st.download_button("Export Forensic Report (JSON)", json_data, "tripwire_forensic_report.json", "application/json", key="paused_export_json", **WIDTH_KWARGS)

if __name__ == "__main__":
    main()
