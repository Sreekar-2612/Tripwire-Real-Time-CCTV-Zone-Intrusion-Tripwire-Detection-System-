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
        self.x = np.array([[init_x], [init_y], [0.0], [0.0]], dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)

    def predict(self) -> Tuple[float, float]:
        """
        Projects state and covariance forward by dt:
        x' = F * x
        P' = F * P * F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
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
        z = np.array([[meas_x], [meas_y]], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        I = np.eye(4, dtype=np.float64)
        # Joseph form for numerical stability: (I - KH)P(I - KH)^T + KRK^T
        I_KH = I - (K @ self.H)
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))
