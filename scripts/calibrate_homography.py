import argparse
import sys
import json
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

from cv_core.geometry.homography import HomographyTransformer

class HomographyCalibratorCLI:
    """
    Interactive 4-Point Ground Calibration Tool.
    Allows a user to click 4 known points on a floor/ground plane and map them
    to real-world metric dimensions (e.g., a 4m x 2m rectangle).
    """
    def __init__(self, source_path: str, output_path: str):
        self.source_path = source_path
        self.output_path = output_path
        self.clicked_points: List[Tuple[int, int]] = []
        self.frame: np.ndarray = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicked_points) < 4:
                self.clicked_points.append((x, y))
                print(f"Point {len(self.clicked_points)} selected at pixel: ({x}, {y})")

    def run(self, length_m: float = 6.0, width_m: float = 3.0):
        # Read source
        if self.source_path.isdigit():
            cap = cv2.VideoCapture(int(self.source_path))
            ret, self.frame = cap.read()
            cap.release()
        elif Path(self.source_path).suffix.lower() in [".jpg", ".png", ".jpeg"]:
            self.frame = cv2.imread(self.source_path)
        else:
            cap = cv2.VideoCapture(self.source_path)
            ret, self.frame = cap.read()
            cap.release()

        if self.frame is None:
            print(f"Error: Could not load frame from '{self.source_path}'")
            return

        window_name = "Homography Ground Plane Calibration (Click 4 corners: TL, TR, BR, BL)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("\n" + "="*70)
        print("HOMOGRAPHY GROUND CALIBRATION INSTRUCTIONS:")
        print(f"Define a real-world rectangle of dimensions: {width_m:.1f}m (width) x {length_m:.1f}m (length)")
        print("Click 4 corners on the floor in clockwise order:")
        print("  1. Top-Left (TL)")
        print("  2. Top-Right (TR)")
        print("  3. Bottom-Right (BR)")
        print("  4. Bottom-Left (BL)")
        print("Press 'c' to calculate Homography, 'r' to reset, 'q' to quit.")
        print("="*70 + "\n")

        while True:
            display = self.frame.copy()
            for idx, pt in enumerate(self.clicked_points):
                cv2.circle(display, pt, 5, (0, 0, 255), -1)
                cv2.putText(display, f"P{idx+1}", (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if len(self.clicked_points) == 4:
                pts = np.array(self.clicked_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(display, "4 Points Selected! Press 'c' to compute.", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('r'):
                self.clicked_points = []
                print("Points reset.")
            elif key == ord('q'):
                print("Calibration cancelled.")
                break
            elif key == ord('c') and len(self.clicked_points) == 4:
                # Real-world metric rectangle corners: (0, length), (width, length), (width, 0), (0, 0)
                # In standard camera ground plane: X=0..width, Y=0..length
                world_pts = [
                    (0.0, length_m),       # TL
                    (width_m, length_m),   # TR
                    (width_m, 0.0),        # BR
                    (0.0, 0.0)             # BL
                ]
                
                transformer = HomographyTransformer.from_point_correspondences(
                    image_points=self.clicked_points,
                    world_points=world_pts,
                    world_bounds=(0.0, 0.0, max(20.0, width_m * 2), max(20.0, length_m * 2))
                )

                error = transformer.compute_reprojection_error(self.clicked_points, world_pts)
                print(f"\n✅ Homography Matrix Computed Successfully!")
                print(f"Mean Reprojection Error: {error:.4f} meters")
                print("Homography Matrix H:\n", transformer.H)

                Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
                transformer.save_calibration(self.output_path)
                print(f"✅ Calibration matrix saved to: {self.output_path}\n")
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Ground Plane Homography for CCTV Camera")
    parser.add_argument("--source", type=str, required=True, help="Path to video file, camera index, or test image")
    parser.add_argument("--output", type=str, default="data/calibration.json", help="Path to save calibration JSON")
    parser.add_argument("--length", type=float, default=6.0, help="Real-world length (depth) in meters of calibrated patch")
    parser.add_argument("--width", type=float, default=3.0, help="Real-world width in meters of calibrated patch")
    args = parser.parse_args()

    calibrator = HomographyCalibratorCLI(args.source, args.output)
    calibrator.run(length_m=args.length, width_m=args.width)
