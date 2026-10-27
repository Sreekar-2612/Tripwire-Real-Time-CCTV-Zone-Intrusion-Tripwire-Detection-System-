import os
import cv2
import numpy as np
from pathlib import Path

def create_synthetic_cctv_video(
    output_path: str,
    duration_sec: int = 10,
    fps: int = 25,
    width: int = 640,
    height: int = 480
):
    """
    Creates a realistic synthetic CCTV video containing a walking person
    moving across the frame and entering/exiting an intrusion zone.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    # Generate background: Surveillance corridor / courtyard
    base_bg = np.zeros((height, width, 3), dtype=np.uint8)
    # Floor (grayish tiles)
    base_bg[int(height * 0.4):, :] = (130, 135, 140)
    for y in range(int(height * 0.4), height, 40):
        cv2.line(base_bg, (0, y), (width, y), (110, 115, 120), 1)
    for x in range(0, width, 60):
        cv2.line(base_bg, (x, int(height * 0.4)), (x, height), (110, 115, 120), 1)

    # Walls
    base_bg[:int(height * 0.4), :] = (70, 75, 80)
    cv2.line(base_bg, (0, int(height * 0.4)), (width, int(height * 0.4)), (50, 55, 60), 2)

    # Simulated walking person path:
    # Starts at left (x = 40), walks through center (x = 320), exits at right (x = 600)
    for f in range(total_frames):
        frame = base_bg.copy()
        
        # Add slight CCTV sensor noise
        noise = np.random.normal(0, 3, (height, width, 3)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Person position
        progress = f / float(total_frames)
        # Walk from x=50 to x=580, y around 320
        px = int(50 + progress * 520)
        # Slight bobbing motion for realistic walking
        py = int(320 + np.sin(f * 0.6) * 4)

        person_h = 130
        person_w = 40

        # Draw realistic human shape (head, torso, legs)
        # Head
        head_radius = 11
        head_center = (px, py - person_h + head_radius)
        cv2.circle(frame, head_center, head_radius, (170, 190, 210), -1)

        # Torso / jacket (dark blue/jacket)
        torso_top = (px - 14, head_center[1] + head_radius)
        torso_bot = (px + 14, py - 45)
        cv2.rectangle(frame, torso_top, torso_bot, (40, 50, 140), -1)

        # Legs (walking stride)
        leg_stride = int(np.sin(f * 0.6) * 12)
        # Left leg
        cv2.line(frame, (px - 7, py - 45), (px - 7 + leg_stride, py), (30, 35, 45), 6)
        # Right leg
        cv2.line(frame, (px + 7, py - 45), (px + 7 - leg_stride, py), (30, 35, 45), 6)

        # CCTV timestamp watermark
        time_text = f"CAM-01 [SYNTH] 2026-10-27 21:30:{f//fps:02d}.{int((f%fps)*(1000/fps)):03d}"
        cv2.putText(frame, time_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        out.write(frame)

    out.release()
    print(f"Generated synthetic CCTV video: {output_path} ({duration_sec}s, {fps}fps)")

if __name__ == "__main__":
    test_video_path = Path(__file__).resolve().parent.parent / "data" / "recordings" / "cctv_sample.mp4"
    create_synthetic_cctv_video(str(test_video_path), duration_sec=10, fps=25)
