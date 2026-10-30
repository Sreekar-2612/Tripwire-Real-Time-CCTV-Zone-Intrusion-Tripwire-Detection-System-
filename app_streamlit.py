import time
from pathlib import Path
import cv2
import numpy as np

# Try importing streamlit
try:
    import streamlit as st
except ImportError:
    st = None

from ultralytics import YOLO
from cv_core.geometry.homography import HomographyTransformer
from cv_core.geometry.zones_3d import MetricZone3D
from cv_core.tracking.tracker import MultiTargetTracker
from cv_core.visualizer.bev_radar import BEVRadarRenderer

@st.cache_resource
def get_detector():
    """Caches YOLO detector in memory across Streamlit reruns."""
    return YOLO("yolov8n.pt")

def main():
    if st is None:
        print("Please install Streamlit using: pip install streamlit")
        return

    st.set_page_config(
        page_title="Tripwire — CCTV Intrusion & BEV Radar",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Session State Initialization
    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    # Custom styling
    st.markdown("""
        <style>
        .main-header { font-size: 26px; font-weight: 700; color: #38bdf8; margin-bottom: 4px; }
        .sub-header { font-size: 14px; color: #94a3b8; margin-bottom: 20px; }
        .stMetric { background-color: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #1e293b; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🛡️ TRIPWIRE: CCTV Intrusion & Bird\'s Eye View (BEV) Radar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-Time Planar Homography, 4D Kalman Tracking & Metric Spatial Intelligence</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Surveillance Controls")
    feed_option = st.sidebar.selectbox(
        "Select Video Feed",
        [
            "Walking Corridor CCTV",
            "Hallway Intrusion",
            "Pedestrian Crossing",
            "Live Laptop Camera (Webcam 0)",
            "External Camera (Webcam 1)",
            "Upload Custom MP4"
        ]
    )

    video_path = None
    if feed_option == "Walking Corridor CCTV":
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
            upload_dir = Path("data/videos")
            upload_dir.mkdir(parents=True, exist_ok=True)
            temp_path = upload_dir / f"uploaded_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path = str(temp_path)

    conf_thresh = st.sidebar.slider("Detection Confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    speed_alarm = st.sidebar.slider("Speed Alarm Threshold (km/h)", min_value=1.0, max_value=25.0, value=5.0, step=0.5)
    show_grid = st.sidebar.checkbox("Project Perspective Ground Grid", value=True)

    st.sidebar.markdown("---")
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("▶️ Start Live", type="primary", use_container_width=True):
            st.session_state.is_running = True
    with col_btn2:
        if st.button("⏹️ Stop Feed", use_container_width=True):
            st.session_state.is_running = False

    st.sidebar.markdown("---")
    st.sidebar.markdown("**CV Algorithms Active:**")
    st.sidebar.markdown("- `YOLOv8-Nano` Person Detector")
    st.sidebar.markdown("- `4D Kalman Filter` (Constant Velocity)")
    st.sidebar.markdown("- `Hungarian Bipartite Matcher`")
    st.sidebar.markdown("- `Planar Homography` DLT Engine")
    st.sidebar.markdown("- `BEV Radar Map` (20m x 20m)")

    # Dynamic Top Metric Banners
    col1, col2, col3, col4 = st.columns(4)
    metric_mota = col1.empty()
    metric_fps = col2.empty()
    metric_tracks = col3.empty()
    metric_breaches = col4.empty()

    # Initial static placeholders
    metric_mota.metric("Detection Confidence", f"{conf_thresh * 100:.0f}%", "Active Filter")
    metric_fps.metric("Processing Speed", "Standby", "0.0 FPS")
    metric_tracks.metric("Active Targets", "0", "Awaiting Feed")
    metric_breaches.metric("Intrusion Breaches", "0", "Secure")

    st.markdown("---")

    # Main Display Area
    dual_view_placeholder = st.empty()
    status_text = st.empty()

    if st.session_state.is_running and video_path is not None:
        if isinstance(video_path, int):
            # DirectShow on Windows for instant webcam opening
            cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            status_text.error(f"❌ Failed to open video source or camera ({video_path}). Please check if your camera is in use by another app.")
            st.session_state.is_running = False
            return

        detector = get_detector()
        tracker = MultiTargetTracker(min_hits=2, max_age=15, iou_threshold=0.3)
        radar = BEVRadarRenderer(radar_size=480, world_extent_m=20.0, grid_step_m=2.0)
        
        transformer = HomographyTransformer()
        calib_file = Path("data/calibration.json")
        if calib_file.exists():
            transformer = HomographyTransformer.load_calibration(calib_file)
        else:
            transformer.set_homography(np.eye(3))

        zones = [
            MetricZone3D("z1", "RESTRICTED VAULT", "polygon", [(3.0, 3.0), (12.0, 3.0), (12.0, 10.0), (3.0, 10.0)]),
            MetricZone3D("tw1", "PERIMETER FENCE", "tripwire", [(0.0, 6.0), (16.0, 6.0)])
        ]

        frame_count = 0
        t0 = time.time()

        try:
            while st.session_state.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    if isinstance(video_path, str) and not video_path.isdigit():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                frame_count += 1

                # Grid Projection
                if show_grid and transformer.is_calibrated:
                    segments = transformer.project_ground_grid_to_image(frame.shape[:2], grid_step_m=2.0)
                    grid_overlay = frame.copy()
                    for pt1, pt2 in segments:
                        cv2.line(grid_overlay, pt1, pt2, (60, 80, 90), 1, cv2.LINE_AA)
                    cv2.addWeighted(grid_overlay, 0.4, frame, 0.6, 0, frame)

                # Detection
                results = detector(frame, classes=[0], conf=conf_thresh, verbose=False)
                detections = []
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            detections.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": conf,
                                "foot_point": ((x1 + x2) / 2.0, float(y2))
                            })

                # Tracking
                active_tracks = tracker.update(detections, homography_transformer=transformer)
                radar_tracks = []
                breach_count = 0

                for trk in active_tracks:
                    w_pos = trk.get("world_pos")
                    is_breach = False
                    if w_pos:
                        for z in zones:
                            if z.zone_type == "tripwire":
                                world_trail = trk.get("world_trail", [])
                                if len(world_trail) >= 2:
                                    crossing = z.check_tripwire_crossing(world_trail[-2], world_trail[-1])
                                    if crossing is not None:
                                        is_breach = True
                                        breach_count += 1
                                        break
                            elif z.contains_point(w_pos[0], w_pos[1]):
                                is_breach = True
                                breach_count += 1
                                break

                    speed_kmh = float(trk.get("speed", 0.0))
                    if speed_kmh > speed_alarm:
                        is_breach = True

                    radar_tracks.append({
                        "track_id": trk["track_id"],
                        "world_pos": w_pos,
                        "world_trail": trk.get("world_trail", []),
                        "speed_kmh": speed_kmh,
                        "vx": trk.get("vx", 0.0),
                        "vy": trk.get("vy", 0.0),
                        "is_breaching": is_breach
                    })

                    x1, y1, x2, y2 = map(int, trk["bbox"])
                    color = (0, 0, 255) if is_breach else (0, 240, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"#{trk['track_id']} | {speed_kmh:.1f}km/h", (x1, max(16, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # BEV Radar
                radar_img = radar.render(zones, radar_tracks)
                dual_view = BEVRadarRenderer.create_dual_view(frame, radar_img)

                # Downsample to 1280x480 for fast WebSocket transmission without DOM saturation
                dual_view_resized = cv2.resize(dual_view, (1280, 480), interpolation=cv2.INTER_AREA)
                dual_view_rgb = cv2.cvtColor(dual_view_resized, cv2.COLOR_BGR2RGB)
                dual_view_placeholder.image(dual_view_rgb, channels="RGB", use_container_width=True)

                elapsed = time.time() - t0
                fps = frame_count / max(0.001, elapsed)

                # Dynamically update live metrics
                metric_mota.metric("Detection Confidence", f"{conf_thresh * 100:.0f}%", "Active Filter")
                metric_fps.metric("Processing Speed", f"{fps:.1f} FPS", "Real-Time" if fps >= 20 else "Processing")
                metric_tracks.metric("Active Targets", f"{len(active_tracks)}", f"{len(active_tracks)} in camera frame")
                metric_breaches.metric("Intrusion Breaches", f"{breach_count}", "ALARM ACTIVE" if breach_count > 0 else "Secure")

                status_text.info(f"⚡ Processing FPS: **{fps:.1f}** | Active Targets: **{len(active_tracks)}** | Breaches Detected: **{breach_count}**")

        finally:
            cap.release()
    elif not st.session_state.is_running:
        status_text.warning("⏸️ Surveillance pipeline is paused. Click **▶️ Start Live** in the sidebar to run.")

if __name__ == "__main__":
    main()
