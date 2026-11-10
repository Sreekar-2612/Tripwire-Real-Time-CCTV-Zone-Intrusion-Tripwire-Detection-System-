import numpy as np
from typing import Tuple, Optional

class KalmanFilter2D:
    """
    Constant-Velocity Linear Kalman Filter for 2D ground/pixel tracking.
    State vector: x = [pos_x, pos_y, vel_x, vel_y]^T  (4x1)
    Measurement vector: z = [pos_x, pos_y]^T           (2x1)
    """
    def __init__(
        self,
        dt: float = 1.0 / 25.0,
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 0.2
    ):
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

        # State transition matrix F (4x4)
        # x_{k} = x_{k-1} + dt * vx_{k-1}
        # y_{k} = y_{k-1} + dt * vy_{k-1}
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Measurement matrix H (2x4)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Measurement noise covariance R (2x2)
        r = self.measurement_noise_std ** 2
        self.R = np.eye(2, dtype=np.float64) * r

        # Discrete White Noise Acceleration (DWNA) Model Q (4x4)
        q = self.process_noise_std ** 2
        dt2 = (self.dt ** 2) / 2.0
        dt3 = (self.dt ** 3) / 2.0
        dt4 = (self.dt ** 4) / 4.0
        self.Q = np.array([
            [dt4, 0.0, dt3, 0.0],
            [0.0, dt4, 0.0, dt3],
            [dt3, 0.0, self.dt ** 2, 0.0],
            [0.0, dt3, 0.0, self.dt ** 2]
        ], dtype=np.float64) * q

        # State estimate x and covariance P
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0

    def initialize(self, init_x: float, init_y: float):
        """Initializes state with first detection coordinates and zero velocity."""
        ix = float(init_x) if np.isfinite(init_x) else 0.0
        iy = float(init_y) if np.isfinite(init_y) else 0.0
        self.x = np.array([[ix], [iy], [0.0], [0.0]], dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)

    def predict(self) -> Tuple[float, float]:
        """
        Projects state and covariance forward by dt:
        x' = F * x
        P' = F * P * F^T + Q
        """
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.P)):
            self.x = np.zeros((4, 1), dtype=np.float64)
            self.P = np.eye(4, dtype=np.float64) * 10.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.P)):
            self.x = np.zeros((4, 1), dtype=np.float64)
            self.P = np.eye(4, dtype=np.float64) * 10.0

        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, meas_x: float, meas_y: float):
        """
        Incorporates new sensor measurement z:
        y = z - H * x
        S = H * P * H^T + R
        K = P * H^T * S^-1
        x = x + K * y
        P = (I - K * H) * P
        """
        if not (np.isfinite(meas_x) and np.isfinite(meas_y)):
            return

        z = np.array([[float(meas_x)], [float(meas_y)]], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        new_x = self.x + (K @ y)
        I = np.eye(4, dtype=np.float64)
        # Joseph form for numerical stability: (I - KH)P(I - KH)^T + KRK^T
        I_KH = I - (K @ self.H)
        new_P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        if np.all(np.isfinite(new_x)) and np.all(np.isfinite(new_P)):
            self.x = new_x
            self.P = new_P

    @property
    def position(self) -> Tuple[float, float]:
        px = float(self.x[0, 0])
        py = float(self.x[1, 0])
        return (px if np.isfinite(px) else 0.0, py if np.isfinite(py) else 0.0)

    @property
    def velocity(self) -> Tuple[float, float]:
        vx = float(self.x[2, 0])
        vy = float(self.x[3, 0])
        return (vx if np.isfinite(vx) else 0.0, vy if np.isfinite(vy) else 0.0)

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        spd = float(np.hypot(vx, vy))
        return spd if np.isfinite(spd) else 0.0
