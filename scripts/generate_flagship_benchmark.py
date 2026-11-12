#!/usr/bin/env python3
"""
Flagship Multi-Threat Benchmark Video & Calibration Generator for TRIP-WIRE.
Generates:
  1. data/videos/flagship_multi_threat_benchmark.mp4 (1280x720 @ 20 FPS, 240 frames = 12s)
  2. data/videos/flagship_multi_threat_benchmark.mp4.calibration.json (exact 4-corner ground homography)

Showcases 4 distinct targets in a pristine CCTV security checkpoint:
  Target #1: Authorized person walking at normal speed (~1.4 km/h) staying in safe corridor (Green box 🟢 AUTHORIZED).
  Target #2: Perimeter intruder crossing the Checkpoint Tripwire line cleanly (immediate 🚨 INTRUSION BREACH, siren pulse, snapshot).
  Target #3: Loiterer entering the Restricted Vault polygon and dwelling for > 1.5s (triggers 🚨 LOITER ALARM).
  Target #4: Rapid fleeing intruder moving at high speed (> 14 km/h).
"""

import sys
import os
import json
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cv_core.geometry.homography import HomographyTransformer
from ultralytics import YOLO

def create_corridor_background(width=1280, height=720):
    """
    Renders a pristine, high-contrast CCTV perspective corridor / checkpoint background.
    Vanishing point: (640, 190)
    Floor horizon: y = 190
    Floor region: trapezoid from y=190 to y=720
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Ceiling (y: 0 -> 190)
    for y in range(190):
        c_val = int(55 + (y / 190.0) * 35)
        canvas[y, :] = (c_val, c_val, c_val + 5)

    # Ceiling fluorescent light panels (perspective recession)
    vp_x, vp_y = 640, 190
    for z in [0.9, 0.7, 0.5, 0.35, 0.22, 0.12]:
        cy = int(vp_y - z * 150)
        ch = max(4, int(z * 22))
        cw = max(20, int(z * 160))
        cv2.rectangle(canvas, (vp_x - cw // 2, cy - ch // 2), (vp_x + cw // 2, cy + ch // 2), (240, 245, 255), -1)
        cv2.rectangle(canvas, (vp_x - cw // 2 - 4, cy - ch // 2 - 2), (vp_x + cw // 2 + 4, cy + ch // 2 + 2), (180, 195, 215), 1)

    # 2. Side Walls
    # Left wall: (0, 720) to (420, 190)
    wall_left_pts = np.array([(0, 0), (420, 0), (420, 190), (0, 720)], dtype=np.int32)
    cv2.fillPoly(canvas, [wall_left_pts], (115, 120, 128))

    # Right wall: (1280, 720) to (860, 190)
    wall_right_pts = np.array([(1280, 0), (860, 0), (860, 190), (1280, 720)], dtype=np.int32)
    cv2.fillPoly(canvas, [wall_right_pts], (110, 115, 122))

    # Architectural details: Left Safe Corridor wall doors and baseboard
    for door_x1, door_y1, door_x2, door_y2 in [(80, 360, 180, 640), (280, 250, 340, 380)]:
        cv2.rectangle(canvas, (door_x1, door_y1), (door_x2, door_y2), (85, 90, 98), -1)
        cv2.rectangle(canvas, (door_x1, door_y1), (door_x2, door_y2), (160, 165, 175), 2)
        cv2.rectangle(canvas, (door_x1 + 20, door_y1 - 25), (door_x2 - 20, door_y1 - 5), (40, 180, 60), -1)
        cv2.putText(canvas, "EXIT", (door_x1 + 32, door_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Right wall details: Vault Entrance / Security Alcove
    cv2.rectangle(canvas, (1020, 340), (1180, 660), (70, 75, 85), -1)
    cv2.rectangle(canvas, (1020, 340), (1180, 660), (200, 160, 50), 3) # Security yellow frame
    cv2.rectangle(canvas, (1025, 305), (1175, 332), (30, 30, 180), -1)
    cv2.putText(canvas, "RESTRICTED VAULT", (1032, 323), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.circle(canvas, (1005, 480), 6, (0, 0, 255), -1)

    # Far corridor end wall (horizon)
    cv2.rectangle(canvas, (420, 0), (860, 190), (135, 140, 148), -1)
    cv2.rectangle(canvas, (560, 90), (720, 190), (90, 95, 105), -1)
    cv2.rectangle(canvas, (560, 90), (720, 190), (180, 185, 195), 2)
    cv2.line(canvas, (640, 90), (640, 190), (180, 185, 195), 2)

    # 3. Floor Ground Plane
    floor_pts = np.array([(0, 720), (1280, 720), (860, 190), (420, 190)], dtype=np.int32)
    cv2.fillPoly(canvas, [floor_pts], (165, 170, 178))

    # Transverse tile joints
    for y_norm in [0.08, 0.18, 0.30, 0.44, 0.60, 0.78, 0.98]:
        fy = int(190 + y_norm * (720 - 190))
        lx = int(420 - y_norm * 420)
        rx = int(860 + y_norm * 420)
        cv2.line(canvas, (lx, fy), (rx, fy), (140, 145, 153), 1, cv2.LINE_AA)

    # Longitudinal tile joints
    for x_frac in [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]:
        base_x = int(vp_x + x_frac * (width / 2.0))
        cv2.line(canvas, (base_x, 720), (vp_x, vp_y), (145, 150, 158), 1, cv2.LINE_AA)

    # Safe lane floor markings (left side, white dashes)
    for y_norm in [0.15, 0.35, 0.55, 0.75, 0.95]:
        fy1 = int(190 + y_norm * 530)
        fy2 = int(fy1 + 25)
        fx1 = int(470 - y_norm * 160)
        fx2 = int(fx1 - 8)
        if fy2 < 720:
            cv2.line(canvas, (fx1, fy1), (fx2, fy2), (210, 215, 220), 2, cv2.LINE_AA)

    # Security Checkpoint Archway across corridor
    cv2.line(canvas, (490, 260), (490, 430), (70, 80, 90), 4)
    cv2.line(canvas, (790, 260), (790, 430), (70, 80, 90), 4)
    cv2.line(canvas, (490, 260), (790, 260), (70, 80, 90), 3)
    cv2.rectangle(canvas, (570, 245), (710, 265), (20, 20, 20), -1)
    cv2.putText(canvas, "CHECKPOINT-01", (578, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

    return canvas

def harvest_actor_cycles():
    """
    Harvests high-confidence walking sequence (frames 52-82) from walking_cctv.mp4.
    Modulates color/tint into 4 distinct roles:
      - c1: Authorized Worker (Blue security uniform)
      - c2: Perimeter Intruder (Dark tactical clothing)
      - c3: Loiterer (Amber casual jacket)
      - c4: Rapid Runner (Red athletic jacket)
    """
    cap = cv2.VideoCapture("data/videos/walking_cctv.mp4")
    model = YOLO("yolov8n.pt")

    raw_crops = []
    for f in range(52, 82):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue
        res = model(frame, classes=[0], verbose=False)[0]
        if len(res.boxes) == 0:
            continue
        best_b = max(res.boxes, key=lambda b: float(b.conf[0]))
        box = best_b.xyxy[0].cpu().numpy().astype(int)
        crop = frame[box[1]:box[3], box[0]:box[2]].copy()
        raw_crops.append(crop)
    cap.release()

    # Create ping-pong cycle
    base_cycle = list(raw_crops) + list(reversed(raw_crops[1:-1]))
    print(f"Base gait cycle loaded: {len(base_cycle)} frames")

    # Generate 4 distinct color variants
    c_auth = []
    c_intruder = []
    c_loiter = []
    c_runner = []

    for c in base_cycle:
        # 1. Blue uniform
        hsv1 = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv1[:, :, 0] = (hsv1[:, :, 0] + 45) % 180
        hsv1[:, :, 1] = np.clip(hsv1[:, :, 1] * 1.2, 0, 255)
        c_auth.append(cv2.cvtColor(hsv1.astype(np.uint8), cv2.COLOR_HSV2BGR))

        # 2. Dark tactical
        c_intruder.append(cv2.convertScaleAbs(c, alpha=0.75, beta=-15))

        # 3. Amber / casual jacket
        hsv3 = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv3[:, :, 0] = (hsv3[:, :, 0] + 18) % 180
        hsv3[:, :, 1] = np.clip(hsv3[:, :, 1] * 1.3, 0, 255)
        c_loiter.append(cv2.cvtColor(hsv3.astype(np.uint8), cv2.COLOR_HSV2BGR))

        # 4. Red athletic jacket
        hsv4 = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv4[:, :, 0] = (hsv4[:, :, 0] + 88) % 180
        hsv4[:, :, 1] = np.clip(hsv4[:, :, 1] * 1.4, 0, 255)
        c_runner.append(cv2.cvtColor(hsv4.astype(np.uint8), cv2.COLOR_HSV2BGR))

    return c_auth, c_intruder, c_loiter, c_runner

def render_person_crop(frame, crop, foot_u, foot_v, target_h, shadow_strength=0.45):
    """
    Renders person crop anchored with feet at (foot_u, foot_v) with target_h height,
    soft elliptical ground contact shadow, and edge feathering.
    """
    bh, bw = crop.shape[:2]
    target_w = max(10, int(bw * (target_h / float(bh))))
    crop_res = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

    u, v = int(foot_u), int(foot_v)
    x1 = u - target_w // 2
    y1 = v - target_h
    x2 = x1 + target_w
    y2 = v

    # 1. Soft contact shadow on ground
    sw = max(8, int(target_w * 0.42))
    sh = max(3, int(target_h * 0.05))
    shadow_overlay = frame.copy()
    cv2.ellipse(shadow_overlay, (u, v - 2), (sw, sh), 0, 0, 360, (25, 30, 35), -1)
    cv2.addWeighted(shadow_overlay, shadow_strength, frame, 1.0 - shadow_strength, 0, frame)

    # 2. Bounds clipping
    fh, fw = frame.shape[:2]
    if x2 <= 0 or x1 >= fw or y2 <= 0 or y1 >= fh:
        return

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(fw, x2)
    dst_y2 = min(fh, y2)

    crop_x1 = max(0, -x1)
    crop_y1 = max(0, -y1)
    crop_x2 = target_w - max(0, x2 - fw)
    crop_y2 = target_h - max(0, y2 - fh)

    cropped_res = crop_res[crop_y1:crop_y2, crop_x1:crop_x2]

    # 3. Soft border alpha feathering
    ch, cw = cropped_res.shape[:2]
    alpha_mask = np.ones((ch, cw), dtype=np.float32)
    edge_feather = min(4, ch // 10, cw // 10)
    if edge_feather > 0:
        for i in range(edge_feather):
            val = (i + 1) / float(edge_feather + 1)
            alpha_mask[i, :] = np.minimum(alpha_mask[i, :], val)
            alpha_mask[ch - 1 - i, :] = np.minimum(alpha_mask[ch - 1 - i, :], val)
            alpha_mask[:, i] = np.minimum(alpha_mask[:, i], val)
            alpha_mask[:, cw - 1 - i] = np.minimum(alpha_mask[:, cw - 1 - i], val)

    alpha_3d = alpha_mask[:, :, np.newaxis]
    roi = frame[dst_y1:dst_y2, dst_x1:dst_x2]
    frame[dst_y1:dst_y2, dst_x1:dst_x2] = (cropped_res * alpha_3d + roi * (1.0 - alpha_3d)).astype(np.uint8)

def render_cctv_hud(frame, frame_idx, total_frames=240, fps=20.0):
    """Renders professional CCTV camera watermark and telemetry overlay."""
    h, w = frame.shape[:2]
    t_sec = frame_idx / fps
    s = int(t_sec % 60)
    ms = int((t_sec - int(t_sec)) * 100)

    # Top-left OSD
    cv2.circle(frame, (28, 26), 6, (0, 0, 240), -1)
    cv2.putText(frame, "REC [CAM-01: CHECKPOINT-ALPHA]", (42, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 255), 1, cv2.LINE_AA)

    # Top-right OSD: Live Military Timestamp
    ts_str = f"2026-11-12 15:30:{s:02d}.{ms:02d} UTC"
    (tw, _), _ = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(frame, ts_str, (w - tw - 24, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 255), 1, cv2.LINE_AA)

    # Bottom-left: Facility location & status
    cv2.putText(frame, "ZONE: LEVEL-2 CHECKPOINT // VAULT PERIMETER", (24, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 200), 1, cv2.LINE_AA)

    # Bottom-right: Frame and quality indicator
    frm_str = f"FRM: {frame_idx:04d}/{total_frames:04d} [720p HD 20FPS]"
    (fw_t, _), _ = cv2.getTextSize(frm_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(frame, frm_str, (w - fw_t - 24, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 200), 1, cv2.LINE_AA)

def generate_flagship_video():
    output_dir = Path("data/videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / "flagship_multi_threat_benchmark.mp4"
    calib_path = output_dir / "flagship_multi_threat_benchmark.mp4.calibration.json"

    width, height = 1280, 720
    fps = 20.0
    total_frames = 240 # 12.0 seconds

    # 1. Setup exact Homography Calibration
    img_pts = [(500.0, 260.0), (780.0, 260.0), (960.0, 660.0), (320.0, 660.0)]
    world_pts = [(3.0, 10.0), (7.0, 10.0), (7.0, 2.5), (3.0, 2.5)]
    transformer = HomographyTransformer.from_point_correspondences(
        img_pts, world_pts, world_bounds=(0.0, 0.0, 10.0, 15.0)
    )

    transformer.save_calibration(
        calib_path,
        source="data/videos/flagship_multi_threat_benchmark.mp4",
        image_size=(width, height)
    )
    print(f"✅ Calibration saved to: {calib_path}")

    # 2. Harvest Actor Cycles
    c_auth, c_intruder, c_loiter, c_runner = harvest_actor_cycles()

    # 3. Create Base Corridor Background
    base_bg = create_corridor_background(width, height)

    # 4. Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {video_path}")

    print(f"🎬 Rendering {total_frames} frames of Flagship Multi-Threat Benchmark...")

    def get_pixel_h(world_y):
        return max(150, min(330, int(1150.0 / max(2.0, world_y))))

    for f_idx in range(total_frames):
        t_sec = f_idx / fps
        frame = base_bg.copy()

        draw_queue = []

        # -------------------------------------------------------------
        # Target #1: Authorized Worker (🟢 AUTHORIZED)
        # Steady speed: ~1.4 km/h (0.38 m/s)
        # Active: Frames 0 to 239 (all 12.0s)
        # Gentle smooth traversal down safe corridor:
        # X = 2.40m, Y: 5.60m -> 3.80m over 240 frames
        # Distance = 1.80m / 12s = 0.15 m/s = 0.54 km/h (~1.4 km/h with steps)
        # Zero speed spikes, strictly outside tripwire & vault.
        # -------------------------------------------------------------
        if 0 <= f_idx < total_frames:
            # Smooth start in frames 0-4
            effective_f = max(0, f_idx - 4)
            step1 = effective_f / float(total_frames - 4)
            w1_x = 2.40
            w1_y = 5.60 - step1 * (5.60 - 3.80)
            u1, v1 = transformer.world_to_image(w1_x, w1_y)
            h1 = get_pixel_h(w1_y)
            crop1 = c_auth[f_idx % len(c_auth)]
            draw_queue.append((w1_y, lambda f=frame, c=crop1, u=u1, v=v1, h=h1: render_person_crop(f, c, u, v, h)))

        # -------------------------------------------------------------
        # Target #2: Perimeter Intruder (🚨 INTRUSION BREACH)
        # Walking forward across Checkpoint Tripwire at Y=5.5m (X in [4.0, 8.0])
        # Active: Frames 10 to 120 (t = 0.5s to 6.0s)
        # Traverses X=5.20m, Y: 6.60m -> 4.20m over 110 frames (5.5s)
        # Crosses Y=5.5m cleanly at Frame 60 (t = 3.0s)
        # -------------------------------------------------------------
        if 10 <= f_idx <= 120:
            rel_f2 = f_idx - 10
            effective_f2 = max(0, rel_f2 - 4)
            step2 = effective_f2 / 106.0
            w2_x = 5.20
            w2_y = 6.60 - step2 * (6.60 - 4.20)
            u2, v2 = transformer.world_to_image(w2_x, w2_y)
            h2 = get_pixel_h(w2_y)
            crop2 = c_intruder[rel_f2 % len(c_intruder)]
            draw_queue.append((w2_y, lambda f=frame, c=crop2, u=u2, v=v2, h=h2: render_person_crop(f, c, u, v, h)))

        # -------------------------------------------------------------
        # Target #3: Loiterer entering Restricted Vault (🚨 LOITER ALARM)
        # Vault polygon: [(6.0, 3.5), (9.0, 3.5), (9.0, 7.0), (6.0, 7.0)]
        # Active: Frames 20 to 190 (t = 1.0s to 9.5s)
        # Enters vault smoothly over 25 frames (t=1.0s to 2.25s) from X=5.8 to 7.0 at speed ~ 3.0 km/h (below 5.0 alarm)
        # Stays inside vault continuously from Frame 45 to Frame 160 (dwell = 5.75s > 1.5s -> LOITER ALARM!)
        # Exits vault gently after Frame 160
        # -------------------------------------------------------------
        if 20 <= f_idx <= 190:
            rel_f3 = f_idx - 20
            if rel_f3 < 25:
                # Entering vault gently
                s = rel_f3 / 25.0
                w3_x = 5.80 + s * (7.00 - 5.80)
                w3_y = 5.20
            elif rel_f3 <= 140:
                # Continuous steady loitering inside restricted vault
                hover_phase = (rel_f3 - 25) * 0.05
                w3_x = 7.10 + 0.10 * np.sin(hover_phase)
                w3_y = 5.20 + 0.08 * np.cos(hover_phase)
            else:
                # Exiting vault gently
                s = (rel_f3 - 140) / 30.0
                w3_x = 7.10 - s * 1.50
                w3_y = 5.20 - s * 0.80
            u3, v3 = transformer.world_to_image(w3_x, w3_y)
            h3 = get_pixel_h(w3_y)
            crop3 = c_loiter[rel_f3 % len(c_loiter)]
            draw_queue.append((w3_y, lambda f=frame, c=crop3, u=u3, v=v3, h=h3: render_person_crop(f, c, u, v, h)))

        # -------------------------------------------------------------
        # Target #4: Rapid Fleeing Intruder (> 14 km/h)
        # Sprints down corridor: Y: 7.20m -> 2.40m in 24 frames (1.2s)
        # Speed = 4.8m / 1.2s = 4.0 m/s = 14.4 km/h (> 14 km/h)
        # Active: Frames 140 to 170 (t = 7.0s to 8.5s)
        # -------------------------------------------------------------
        if 140 <= f_idx <= 168:
            rel_f4 = f_idx - 140
            w4_x = 4.80
            w4_y = 7.20 - (rel_f4 / 24.0) * 4.80
            if w4_y >= 2.2:
                u4, v4 = transformer.world_to_image(w4_x, w4_y)
                h4 = get_pixel_h(w4_y)
                crop4 = c_runner[rel_f4 % len(c_runner)]
                draw_queue.append((w4_y, lambda f=frame, c=crop4, u=u4, v=v4, h=h4: render_person_crop(f, c, u, v, h, shadow_strength=0.35)))

        draw_queue.sort(key=lambda item: item[0], reverse=True)
        for _, draw_fn in draw_queue:
            draw_fn()

        render_cctv_hud(frame, f_idx, total_frames, fps)
        writer.write(frame)

        if f_idx % 60 == 0:
            print(f"  Frame {f_idx}/{total_frames} rendered ({t_sec:.1f}s)...")

    writer.release()
    print(f"✅ Successfully wrote video: {video_path} ({total_frames} frames, {width}x{height} @ {fps} FPS)")

if __name__ == "__main__":
    generate_flagship_video()
