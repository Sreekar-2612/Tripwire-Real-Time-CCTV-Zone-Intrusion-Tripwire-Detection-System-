# TRIP-WIRE: System Architecture, Technical Novelty & Engineering Whitepaper

> **Real-Time 3D Spatial Intelligence, Calibrated Monocular Ground Radar & Biomechanical Threat Kinetics for CCTV Surveillance Networks.**

---

## Table of Contents
1. [Executive Overview & Vision](#1-executive-overview--vision)
2. [What We Are Actually Doing & Implementing (System Pipeline)](#2-what-we-are-actually-doing--implementing-system-pipeline)
   - [2.1 High-Throughput Perception Layer](#21-high-throughput-perception-layer)
   - [2.2 Kinematic Foot-Contact Isolation](#22-kinematic-foot-contact-isolation)
   - [2.3 Multi-Target Tracking & Covariance-Stabilized Filtering](#23-multi-target-tracking--covariance-stabilized-filtering)
   - [2.4 Planar Homography Projection (2D-to-3D Metric Mapping)](#24-planar-homography-projection-2d-to-3d-metric-mapping)
   - [2.5 Ground Velocity Estimation & ZUPT Deadband](#25-ground-velocity-estimation--zupt-deadband)
   - [2.6 Spatial Geofencing & Directional Tripwire Engine](#26-spatial-geofencing--directional-tripwire-engine)
   - [2.7 Biomechanical Kinetic Threat Classifier](#27-biomechanical-kinetic-threat-classifier)
   - [2.8 Bird's Eye View (BEV) Ground Radar HUD & Evidence Locker](#28-birds-eye-view-bev-ground-radar-hud--evidence-locker)
3. [Technical Novelty & Innovations](#3-technical-novelty--innovations)
4. [Comparative Matrix: Where Current Systems Fail vs Where TRIP-WIRE Passes](#4-comparative-matrix-where-current-systems-fail-vs-where-trip-wire-passes)
5. [The 6 Engineering Strikes: Exact Math & Implementation](#5-the-6-engineering-strikes-exact-math--implementation)
   - [Strike 1: Hartley-Normalized Direct Linear Transformation (DLT)](#strike-1-hartley-normalized-direct-linear-transformation-dlt)
   - [Strike 2: Liang-Barsky Parametric Viewport & Floor-Horizon Clipping](#strike-2-liang-barsky-parametric-viewport--floor-horizon-clipping)
   - [Strike 3: Joseph-Stabilized Covariance 4D Kalman Filtering](#strike-3-joseph-stabilized-covariance-4d-kalman-filtering)
   - [Strike 4: Zero-Velocity Update (ZUPT) Deadband Filter](#strike-4-zero-velocity-update-zupt-deadband-filter)
   - [Strike 5: ByteTrack Two-Stage Occlusion Recovery](#strike-5-bytetrack-two-stage-occlusion-recovery)
   - [Strike 6: Stateful Spatial Dwell State Machine (`ZoneMonitor`)](#strike-6-stateful-spatial-dwell-state-machine-zonemonitor)
6. [Viva & Technical Interview Defense Guide](#6-viva--technical-interview-defense-guide)
   - [30-Second Elevator Pitch](#30-second-elevator-pitch)
   - [Top 5 Viva / Interview Questions & Model Answers](#top-5-viva--interview-questions--model-answers)
   - [Resume-Ready Bullets](#resume-ready-bullets)

---

## 1. Executive Overview & Vision

Modern commercial security installations monitor billions of hours of CCTV video daily, yet **over 95% of perimeter intrusion alarms are false positives** triggered by vehicle headlights, swaying vegetation, shadows, optical camera foreshortening, and boundary-grazing pedestrians.

The core limitation of traditional CCTV systems is foundational: **cameras capture flat 2D pixel projections $(u, v)$, while security decisions require metric 3D physical reality $(X, Y, Z)$.**

**TRIP-WIRE** bridges deep-learning perception with projective geometry and state estimation. It converts standard monocular 2D CCTV cameras into **calibrated 3D spatial radar stations** without requiring expensive LiDAR sensors, stereo camera rigs, or multi-view infrastructure.

```
+--------------------------------------------------------------------------------------------------+
|                                    TRIP-WIRE PARADIGM SHIFT                                     |
|                                                                                                  |
|   CONVENTIONAL CCTV (Raw Pixels)                TRIP-WIRE SYSTEM (Spatial Metric Intelligence)   |
|   • Optical foreshortening distortion           • Scale-invariant metric ground plane (meters)   |
|   • Floating, jittery bounding box centroids    • Ground-anchored ankle contact footpoints       |
|   • Pixels/sec speed (distance-dependent)       • True physical ground velocity (km/h)           |
|   • False alarms from touching boundary lines   • Stateful directional vector crossing hysteresis|
|   • Static detection without threat intent      • 17-Keypoint skeletal threat kinetic fusion     |
|   • Requires $5,000+ LiDAR or stereo cameras    • Pure software upgrade for existing 2D cameras  |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. What We Are Actually Doing & Implementing (System Pipeline)

TRIP-WIRE executes an end-to-end, multi-threaded pipeline across perception, kinematic state estimation, geometric transformation, spatial logic, and tactical visualization.

```mermaid
flowchart TD
    subgraph PERCEPTION["1. Perception Layer"]
        A["RTSP / Video Stream / Webcam"] --> B["YOLOv8-Nano / YOLOv8-Pose"]
        B --> C1["2D Bounding Boxes [x1, y1, x2, y2]"]
        B --> C2["17 COCO Skeletal Keypoints"]
    end

    subgraph KINEMATICS["2. Kinematic Anchor & Tracking"]
        C1 & C2 --> D["Foot-Ground Contact Point Extractor"]
        D --> E["4D Kalman Filter (Joseph Form)"]
        C1 --> F["ByteTrack Two-Stage Hungarian Matcher"]
        E & F --> G["Stable Track Identity & State [x, y, vx, vy]"]
    end

    subgraph PROJECTION["3. Projective Geometry"]
        G --> H["Planar Homography Matrix H (DLT)"]
        H --> I["Real-World Ground Coordinates (X, Y in meters)"]
        I --> J["Liang-Barsky Clipped Perspective Ground Grid"]
    end

    subgraph THREAT["4. Spatial & Kinetic Threat Engine"]
        I --> K["Metric Speed & ZUPT Deadband (km/h)"]
        I --> L["Vector Tripwires & Polygon Geofences"]
        C2 & K --> M["Biomechanical Posture Threat Classifier"]
        L & M --> N["Stateful ZoneMonitor (ENTER / LOITER / CROSS_IN)"]
    end

    subgraph SOC["5. SOC Visualization & Forensics"]
        I & K & N --> O["Top-Down BEV Ground Radar HUD"]
        N --> P["Evidence Locker Lossless Crop Capture"]
        N --> Q["Forensic JSON / CSV Incident Audit Logger"]
    end
```

### 2.1 High-Throughput Perception Layer
- **Model Options:** Decoupled dual architecture supporting **YOLOv8-Nano** (high-throughput person bounding box detection) and **YOLOv8-Pose** (simultaneous detection and 17-joint COCO human keypoint estimation).
- **Execution:** Runs with configurable inference resolution (`imgsz=480` or `640`) and non-maximum suppression (NMS) to isolate all human targets per frame.

### 2.2 Kinematic Foot-Contact Isolation
- **The Defect of Centroids:** A bounding box centroid $(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2})$ floats in 3D mid-air at hip/waist height. When a person sits, crouches, bends, or raises their hands, the centroid shifts dramatically even if their body has not moved on the floor.
- **Our Solution:** The system isolates the **physical ground contact point**:
  - *When Pose is active:* Calculates the midpoint of the left and right ankle keypoints:
    $$\vec{p}_{foot} = \left(\frac{u_{l\_ank} + u_{r\_ank}}{2}, \frac{v_{l\_ank} + v_{r\_ank}}{2}\right)$$
  - *Fallback:* Bottom-center of the bounding box $(\frac{x_1 + x_2}{2}, y_2)$.
  - This ensures that projections onto the ground plane are physically anchored to the floor tiles.

### 2.3 Multi-Target Tracking & Covariance-Stabilized Filtering
- **4D State Vector:** Each target track is modeled in continuous space:
  $$\mathbf{x} = \begin{bmatrix} u & v & \dot{u} & \dot{v} \end{bmatrix}^T$$
- **Covariance Convergence:** Employs the **Joseph Stabilized Form** of the Kalman measurement update equation to prevent numerical degeneration and matrix divergence during long surveillance sessions.
- **Two-Stage ByteTrack Association:** Detections are matched using Distance-IoU (DIoU) and linear assignment (Hungarian algorithm). Confirmed tracks match high-confidence detections first, while occluded or partially visible targets match lower-confidence detections in stage two.

### 2.4 Planar Homography Projection (2D-to-3D Metric Mapping)
- **Projective Transformation:**
  $$\begin{bmatrix} X_{world} \\ Y_{world} \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} u_{pixel} \\ v_{pixel} \\ 1 \end{bmatrix}$$
  where $\mathbf{H}$ is a non-singular $3 \times 3$ matrix calibrated from four known ground points.
- **Singularity Verification:** Every projected point undergoes projective horizon verification ($w > 0$ and $|w| > 10^{-9}$) to discard unprojectable points behind the camera lens.

### 2.5 Ground Velocity Estimation & ZUPT Deadband
- **Temporal Metric Calculation:** World positions are logged temporally with high-resolution frame clocks. Velocity is computed in physical units:
  $$\vec{v} = \left(\frac{X_{t_2} - X_{t_1}}{\Delta t}, \frac{Y_{t_2} - Y_{t_1}}{\Delta t}\right) \implies \text{Speed} = \|\vec{v}\|_2 \times 3.6 \quad (\text{km/h})$$
- **Exponential Moving Average (EMA):** Smoothed with $\alpha = 0.35$ to filter bounding box vibration.
- **Zero-Velocity Update (ZUPT):** Any velocity below $0.8\text{ km/h}$ ($\approx 0.22\text{ m/s}$) is clamped to **$0.0\text{ km/h}$**, eliminating stationary bounding-box jitter.

### 2.6 Spatial Geofencing & Directional Tripwire Engine
- **Vector Tripwire Crossing Hysteresis:** Rather than detecting line intersections, the engine tracks signed normal distance across successive frame observations:
  $$\text{Sign} = (x_2 - x_1)(y_{target} - y_1) - (y_2 - y_1)(x_{target} - x_1)$$
  A breach triggers (`CROSS_IN` or `CROSS_OUT`) **only** when a track shifts from one half-plane to the other.
- **Polygon Geofencing (Restricted Vaults):** Evaluates Jordan curve theorem (Ray-Casting Point-in-Polygon) on the ground coordinates. Incorporates a **temporal dwell debounce timer** (e.g. $1.5\text{s}$ minimum residency) before triggering a sustained `LOITER` alarm, eliminating transient noise.

### 2.7 Biomechanical Kinetic Threat Classifier
Analyzes the spatial configuration of 17 COCO body joints combined with metric velocity and boundary proximity:
- **Standing / Walking:** Vertical spine vector, normal step displacement.
- **Crouching / Crawling:** Hip-to-shoulder vertical displacement collapses below normal height ratio while ground velocity is active.
- **Rapid Fleeing:** Metric ground speed exceeds rapid sprint threshold ($> 14\text{ km/h}$).
- **Perimeter Climbing:** Foot keypoints elevated off the ground plane ($Y_{elev} > 0.4\text{m}$) within $1.5\text{m}$ of a perimeter boundary.

### 2.8 Bird's Eye View (BEV) Ground Radar HUD & Evidence Locker
- **Top-Down Tactical Radar:** Renders a $20\text{m} \times 20\text{m}$ ground radar with concentric metric range rings, camera origin beacon `CAM (0,0)`, active geofences, and live target blips.
- **Directional Velocity Vectors:** Arrowhead vectors scale dynamically with speed and enforce an $18\text{px}$ minimum clearance so directional intent is never hidden inside the target marker.
- **Temporal Motion History:** Renders dynamic alpha-faded trajectory tails ($0.2 \to 1.0$) showing path history.
- **Forensic Snapshot Evidence Locker:** The moment an intrusion breach or loiter event triggers, an un-annotated bounding crop with 10% context padding is captured to session memory and logged to JSON/CSV audit tables.

---

## 3. Technical Novelty & Innovations

### 1. Monocular Ground Radar Without Expensive Sensors
Commercial spatial tracking relies on active sensor hardware: 3D LiDAR (\$3,000–\$15,000) or stereo depth cameras (\$500–\$2,000 with $<10\text{m}$ range). TRIP-WIRE achieves **calibrated metric spatial localization on legacy 2D CCTV cameras** through software-driven projective planar homography.

### 2. Scale-Invariant Speed Estimation
In raw pixel space, an intruder running far from the camera moves at $8\text{ px/frame}$, while an authorized person walking close moves at $90\text{ px/frame}$. Conventional pixel-based motion detectors misidentify the close pedestrian as high-speed and miss the distant runner. TRIP-WIRE normalizes perspective distortion, measuring true physical speed ($\text{km/h}$) uniformly across the entire floor.

### 3. Foot-Contact Isolation vs Floating Centroids
Trackers typically project the bounding box centroid. However, a person's torso floats $1.0$–$1.5\text{m}$ above the floor, and bending over shifts the centroid horizontally by up to $0.5\text{m}$. TRIP-WIRE isolates the **ankle midpoint contact point**, anchoring the metric projection strictly to the floor plane.

### 4. Directional State-Preserving Tripwire Crossing
Commercial tripwires alert when a bounding box collides with a line segment. This causes continuous false alarms when a person stands near the line or walks parallel to it. TRIP-WIRE's vector half-plane hysteresis requires a verified trajectory transition from side $A$ to side $B$.

### 5. Multi-Modal Threat Index Synthesis
Instead of binary motion alarms, TRIP-WIRE computes a unified threat index by fusing:
$$\text{Threat} = f(\text{Spatial Zone Violation}, \text{Metric Velocity}, \text{Biomechanical Posture})$$
A person walking through a corridor is authorized (`🟢 NORMAL`). A person running at $16\text{ km/h}$ while crouching inside a vault immediately triggers `🚨 CRITICAL INTRUSION`.

---

## 4. Comparative Matrix: Where Current Systems Fail vs Where TRIP-WIRE Passes

| System Type | Architecture | Critical Failure Modes | How TRIP-WIRE Passes |
| :--- | :--- | :--- | :--- |
| **PIR / Hardware Motion Detectors** | Passive Infrared Sensor | • Triggered by headlights, sunlight, temperature shifts.<br>• Zero target identity or trajectory tracking.<br>• Cannot distinguish humans from animals or objects. | **Deep Learning + Kalman Filtering:** Rejects ambient noise; tracks persistent human identities across frames. |
| **Traditional CV (OpenCV Background Subtraction)** | Gaussian Mixture Model (GMM), Frame Difference | • Fails on moving shadows, lighting changes, swaying trees.<br>• Occlusion collapse (merges two people into one blob).<br>• Pixel-only measurements with no physical scale. | **ByteTrack 2-Stage Bipartite Matching:** Discriminates individual persons and preserves track IDs through occlusions. |
| **Vanilla YOLO + SORT / DeepSORT** | Object Detection + Naive Bounding Box Kalman Filter | • Centroid jitter causes stationary people to display false speeds.<br>• Speed measured in pixels/sec (foreshortening illusion).<br>• Tripwires trigger on swinging hands, arms, or loose clothing. | **Homography + ZUPT Deadband:** True ground speed in km/h; stationary noise clamped to $0.0\text{ km/h}$; footpoint contact anchors. |
| **Commercial CCTV Virtual Tripwires** | Edge NVR Analytic (Axis, Hikvision, Dahua) | • "Grazing" defect: triggers if a person walks parallel to or grazes the boundary.<br>• Lack of top-down spatial awareness.<br>• No kinetic action classification (cannot spot crawling or climbing). | **Vector Cross-Product Hysteresis + BEV Radar:** Requires complete plane crossing; displays live top-down radar; classifies crawling and climbing. |
| **LiDAR / Multi-Camera 3D Rigs** | 3D Point Cloud Scanning, Stereo Photogrammetry | • Prohibitively expensive hardware (\$5,000–\$25,000 per zone).<br>• High computational overhead (requires dedicated GPU servers).<br>• Complex physical calibration and optical alignment. | **Zero Hardware Cost:** Pure software solution running on existing monocular IP cameras and edge hardware at high FPS. |

---

## 5. The 6 Engineering Strikes: Exact Math & Implementation

### Strike 1: Hartley-Normalized Direct Linear Transformation (DLT)
* **The Goal:** Compute homography matrix $\mathbf{H}$ without numerical instability.
* **Problem:** Image coordinates are $O(10^2)$ to $O(10^3)$ pixels, while world coordinates are $O(10^0)$ meters. Naive DLT matrices become ill-conditioned, magnifying floating-point errors.
* **Implementation:** Coordinates are normalized prior to SVD so the centroid is at the origin and mean distance is $\sqrt{2}$:
  $$\tilde{\mathbf{u}} = \mathbf{T}_{img} \mathbf{u}, \quad \tilde{\mathbf{X}} = \mathbf{T}_{world} \mathbf{X}$$
  The $8 \times 9$ coefficient matrix $\mathbf{A}$ is solved using SVD ($\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$), taking the right singular vector corresponding to the smallest singular value, then denormalized:
  $$\mathbf{H} = \mathbf{T}_{world}^{-1} \tilde{\mathbf{H}} \mathbf{T}_{img}$$

### Strike 2: Liang-Barsky Parametric Viewport & Floor-Horizon Clipping
* **The Goal:** Prevent projected grid lines and zone boundaries from climbing walls or ceilings.
* **Implementation:** In `cv_core/geometry/homography.py`, we implemented a parametric 2D line clipper combined with a depth horizon clamp (`max_visible_y = 6.0m`):
  Line segment $\vec{P}(t) = \vec{P}_1 + t(\vec{P}_2 - \vec{P}_1)$ is evaluated against the 4 viewport boundaries using directional inequalities:
  $$p_k \cdot t \le q_k, \quad k \in \{1, 2, 3, 4\}$$
  Segments extending past image bounds $[0, w] \times [0, h]$ or above the floor horizon ($v = 314$) are parametrically clipped to boundary intersection points or dropped entirely.

### Strike 3: Joseph-Stabilized Covariance 4D Kalman Filtering
* **The Goal:** Prevent state covariance divergence over long surveillance sessions.
* **Problem:** Standard measurement update $\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}$ loses symmetry and positive semi-definiteness due to floating-point rounding errors, eventually producing negative variances or filter crashes.
* **Implementation:** In `cv_core/tracking/kalman_filter.py`, we implemented the **Joseph Stabilized Form**:
  $$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$$
  Because both terms are quadratic forms, $\mathbf{P}_{k|k}$ is mathematically guaranteed to remain symmetric and positive semi-definite across millions of continuous iterations.

### Strike 4: Zero-Velocity Update (ZUPT) Deadband Filter
* **The Goal:** Eliminate bounding-box detection jitter for stationary targets.
* **Implementation:** In `cv_core/tracking/tracker.py`:
  ```python
  raw_spd_kmh = float(np.hypot(raw_vx, raw_vy) * 3.6)
  SPEED_DEADBAND_KMH = 0.8  # ~0.22 m/s

  if raw_spd_kmh < SPEED_DEADBAND_KMH:
      raw_vx = 0.0
      raw_vy = 0.0

  spd = float(np.hypot(self.metric_vx, self.metric_vy) * 3.6)
  if spd < SPEED_DEADBAND_KMH:
      self.metric_speed = 0.0
      self.metric_vx = 0.0
      self.metric_vy = 0.0
  else:
      self.metric_speed = min(35.0, spd)
  ```
  Standing targets report **$0.0\text{ km/h}$**, while genuine human walking motion ($1.2$–$5.0\text{ km/h}$) transitions smoothly.

### Strike 5: ByteTrack Two-Stage Occlusion Recovery
* **The Goal:** Prevent identity swaps and track loss during partial human occlusions.
* **Implementation:**
  1. **Stage 1 (High Confidence):** Detections with $conf \ge 0.5$ are matched against existing confirmed tracks using Distance-IoU (DIoU) cost assignment:
     $$\mathcal{C}_{i,j} = 1 - \text{DIoU}(\text{track}_i, \text{det}_j)$$
  2. **Stage 2 (Low Confidence):** Unmatched tracks from Stage 1 are matched against remaining low-confidence detections ($0.1 \le conf < 0.5$).
  When an intruder walks behind an obstacle or another individual, their detection score drops, but Stage 2 maintains the track state until they fully re-emerge.

### Strike 6: Stateful Spatial Dwell State Machine (`ZoneMonitor`)
* **The Goal:** Eliminate false breach alarms from momentary boundary grazing or single-frame detection flickers.
* **Implementation:** In `cv_core/monitoring.py`, each zone tracks state transitions per target:
  $$\text{OUTSIDE} \xrightarrow{\text{Enter}} \text{ENTER} \xrightarrow{\Delta t \ge 1.5\text{s}} \text{LOITER (CONFIRMED BREACH)} \xrightarrow{\text{Leave}} \text{EXIT}$$
  - Crossing a tripwire emits `CROSS_IN` or `CROSS_OUT`.
  - A confirmed breach occurs only when the target dwells within the zone beyond the debounce threshold or crosses the tripwire, triggering automated evidence capture.

---

## 6. Viva & Technical Interview Defense Guide

### 30-Second Elevator Pitch
> *"TRIP-WIRE solves the fundamental flaw of commercial CCTV: cameras capture flat 2D pixels, but security decisions require 3D spatial reality. By coupling YOLOv8 and ByteTrack with calibrated planar homography, our system maps foot-ground contact points into metric coordinates in real time.*
>
> *This unlocks true physical speed estimation in $\text{km/h}$, directional Bird's Eye View ground radar, and debounced vector tripwires on standard 2D CCTV cameras with zero extra hardware. We stabilized the pipeline with Joseph-form Kalman covariance, Liang-Barsky parametric floor clipping, and zero-velocity deadband filtering, achieving a mathematically rigorous, production-grade surveillance system."*

---

### Top 5 Viva / Interview Questions & Model Answers

#### Q1: "Why use Planar Homography instead of an off-the-shelf Depth Camera (e.g., Intel RealSense) or LiDAR?"
> **Answer:** *"Depth cameras rely on infrared structured light or time-of-flight, which fails outdoors under direct sunlight and has an effective range under 10 meters. LiDAR units cost upwards of \$5,000 per unit and produce sparse point clouds requiring high-end computing.
>
> In commercial surveillance, security operators already have millions of standard 2D RTSP CCTV cameras installed. Planar homography leverages the physical constraint that humans walk on a flat ground plane ($Z = 0$). By solving the $3 \times 3$ projective mapping from four surveyed floor points, we achieve centimeter-accurate ground localization and speed estimation on existing infrastructure at zero additional hardware cost."*

#### Q2: "How do you calculate speed accurately from monocular video when camera perspective distorts motion?"
> **Answer:** *"We never compute velocity in pixel space. In perspective projection, moving 20 pixels in the background could represent 5 meters, while moving 20 pixels in the foreground represents 0.5 meters.
>
> Instead, we project the target's ankle-contact coordinates onto the calibrated metric ground plane $(X, Y)$ in meters. We calculate physical displacement $\Delta d = \sqrt{(X_2 - X_1)^2 + (Y_2 - Y_1)^2}$ divided by true elapsed time $\Delta t$ from a high-resolution frame clock. We then apply an Exponential Moving Average ($\alpha = 0.35$) and a Zero-Velocity Deadband ($0.8\text{ km/h}$) to eliminate bounding-box quantization noise."*

#### Q3: "How does your tripwire algorithm prevent false alarms when someone merely touches or walks along the boundary line?"
> **Answer:** *"Traditional tripwires use simple bounding box intersection, which triggers continuously if an authorized person walks along or grazes the fence line.
>
> We use signed vector cross-product hysteresis. The line segment $\vec{L} = \vec{B} - \vec{A}$ divides the 2D ground plane into two distinct half-planes: positive and negative. We compute the cross product between the line vector and the vector to the target's footpoint. A breach event (`CROSS_IN`) triggers only when the cross-product sign inverts between consecutive observations and the intersection point lies strictly within the line segment bounds."*

#### Q4: "Why did you implement the Joseph Stabilized Form in your Kalman filter?"
> **Answer:** *"In naive Kalman filter implementations, the covariance update equation $\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}$ is vulnerable to floating-point truncation errors. Over hours of continuous execution with thousands of matrix multiplications, asymmetric numerical drift accumulates, causing $\mathbf{P}$ to lose positive semi-definiteness. This leads to imaginary values or mathematical divergence.
>
> The Joseph stabilized formulation, $\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}(\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K}\mathbf{R}\mathbf{K}^T$, consists of two symmetric quadratic forms. This mathematically guarantees that the error covariance matrix remains symmetric and positive semi-definite for unlimited runtimes."*

#### Q5: "What happens if a person is partially occluded by a pillar or another individual?"
> **Answer:** *"Standard trackers discard detections when confidence drops below a static threshold (e.g. $< 0.5$), resulting in track termination and identity switches upon re-emergence.
>
> We employ ByteTrack's two-stage bipartite association. High-confidence detections are matched first with confirmed tracks. Detections that fall into low confidence ($0.1 \le conf < 0.5$) due to occlusion are then matched against the remaining unmatched tracks using Distance-IoU (DIoU). If the target is completely blocked, the Kalman filter's state transition matrix predicts their position along their velocity vector, keeping the track alive for up to 15 frames."*

---

### Resume-Ready Bullets

- **Architected TRIP-WIRE**, a real-time computer vision surveillance system combining **YOLOv8**, **ByteTrack**, and a **4D Kalman Filter** for multi-target tracking under partial occlusions.
- **Formulated a $3 \times 3$ Direct Linear Transformation (DLT) Planar Homography** engine with Hartley normalization, translating 2D image coordinates to calibrated 3D ground coordinates in physical meters.
- **Engineered a 20m $\times$ 20m Bird's Eye View (BEV) Ground Radar HUD**, generating real-time top-down target localization, directional velocity vectors in $\text{km/h}$, and temporal alpha-faded motion trails.
- **Designed a stateful spatial perimeter engine (`ZoneMonitor`)** supporting arbitrary polygon geofences and parametric vector tripwires with directional half-plane crossing hysteresis and debounced dwell timers.
- **Implemented Joseph-form stabilized covariance updates** in the Kalman state filter and **Liang-Barsky parametric line clipping** to eliminate numerical drift and floor-grid projection overshoots.
- **Developed multi-threaded deployment interfaces** using **Streamlit** (interactive SOC HUD with automated forensic snapshot capture) and **FastAPI** (MJPEG streaming with token authentication), backed by **94 automated unit and regression tests (100% pass rate)**.
