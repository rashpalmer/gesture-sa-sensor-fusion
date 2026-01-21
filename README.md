# Situational Awareness via Sensor Fusion and Gesture Recognition
## Using iPhone Sensor Data (MATLAB Mobile)

---

# PART A: PRODUCT REQUIREMENTS DOCUMENT (PRD)

## 1. Problem Statement & Motivation

### What is Situational Awareness in This Context?

Situational awareness, in this system, refers to the device's ability to understand:
1. **Its own motion state**: Is it stationary, rotating, translating, or executing a complex motion?
2. **Orientation relative to the world**: Which way is "up"? Which direction is the device facing?
3. **User intent interpretation**: What gesture did the user perform, and what action should follow?

This three-layer understanding transforms raw sensor streams into actionable intelligence.

### Why Sensor Fusion is Required

Individual sensors have fundamental limitations:
- **Gyroscope**: Excellent short-term angular rate measurement, but integrating angular rate to get orientation causes unbounded drift over time (typically 1-10°/minute for MEMS gyros)
- **Accelerometer**: Provides gravity direction reference (stable long-term), but corrupted by linear acceleration during motion; also high-frequency noise
- **Magnetometer**: Provides heading reference, but highly susceptible to local magnetic disturbances (steel structures, electronics, etc.)

**Sensor fusion** combines these complementary sources:
- Gyro provides high-frequency dynamics (what's changing NOW)
- Accelerometer/magnetometer provide low-frequency corrections (where should the estimate converge)
- The Extended Kalman Filter optimally weights these based on their noise characteristics

### What is a "Gesture"?

A **gesture** is a short, intentional motion pattern performed by the user with the device. Characteristics:
- Duration: typically 0.3–2.0 seconds
- Contains distinctive kinematic signatures (rotations, accelerations)
- Bounded by periods of relative stillness or steady motion
- Maps to a semantic label (e.g., "flip up", "shake", "twist")

---

## 2. Goals (Must-Haves)

| ID | Requirement | Priority |
|----|-------------|----------|
| G1 | Accept iPhone sensor logs from MATLAB Mobile (CSV or MAT) | Must |
| G2 | Produce time-aligned, calibrated IMU streams with proper units | Must |
| G3 | Estimate attitude/orientation using quaternions | Must |
| G4 | Implement complementary filter (baseline comparison) | Should |
| G5 | Implement linear Kalman Filter for velocity/position estimation | Must |
| G6 | Implement Extended Kalman Filter for quaternion attitude | Must |
| G7 | Gesture segmentation pipeline (detect start/end) | Must |
| G8 | Feature extraction from segmented gestures | Must |
| G9 | Rule-based gesture classification | Must |
| G10 | ML-ready baseline classifier (kNN/SVM stub) | Must |
| G11 | Clear diagnostic plots and performance metrics | Must |

---

## 3. Non-Goals (Out of Scope)

- Camera-based visual-inertial odometry (VIO)
- SLAM or map building
- GPS/GNSS integration
- External beacon positioning (UWB, BLE)
- Production deployment or app development
- Real-time streaming (batch processing only)
- Multi-device fusion

---

## 4. Users & Use Cases

### Primary User
Engineering/CS student demonstrating understanding of sensor fusion and situational awareness concepts.

### Use Cases

**UC1: Educational Demo**
> "I want to record myself doing gestures with MATLAB Mobile, then run the pipeline to see how quaternions and Kalman filters work."

**UC2: Algorithm Comparison**
> "I want to compare EKF vs complementary filter accuracy on the same dataset."

**UC3: ML Prototyping**
> "I want to train a simple classifier on labeled gesture data and evaluate accuracy."

---

## 5. System Overview / Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────────────────┐  │
│  │ MATLAB Mobile│───>│ read_phone_data │───>│  data struct (raw)        │  │
│  │ (iPhone)     │    │                 │    │  .t, .acc, .gyr, .mag     │  │
│  └──────────────┘    └─────────────────┘    └───────────────────────────┘  │
│                                                       │                     │
│                                                       v                     │
│                              ┌─────────────────────────────────────────┐    │
│                              │         preprocess_imu                  │    │
│                              │  - resample to fixed Fs                 │    │
│                              │  - bias removal, filtering              │    │
│                              │  - magnetometer calibration             │    │
│                              └─────────────────────────────────────────┘    │
│                                               │                             │
│                     ┌─────────────────────────┼─────────────────────────┐   │
│                     v                         v                         v   │
│  ┌─────────────────────────┐  ┌──────────────────────┐  ┌────────────────┐ │
│  │ complementary_filter    │  │  ekf_attitude_quat   │  │ kf_linear_motion│ │
│  │ (baseline)              │  │  (quaternion+bias)   │  │ (velocity/pos) │ │
│  └─────────────────────────┘  └──────────────────────┘  └────────────────┘ │
│                     │                         │                         │   │
│                     └─────────────────────────┼─────────────────────────┘   │
│                                               v                             │
│                              ┌─────────────────────────────────────────┐    │
│                              │         segment_gesture                 │    │
│                              │  - energy thresholding                  │    │
│                              │  - hysteresis state machine             │    │
│                              └─────────────────────────────────────────┘    │
│                                               │                             │
│                                               v                             │
│                              ┌─────────────────────────────────────────┐    │
│                              │         extract_features                │    │
│                              │  - RMS, peaks, duration, dominant axis  │    │
│                              │  - zero-crossings, energy distribution  │    │
│                              └─────────────────────────────────────────┘    │
│                                               │                             │
│                          ┌────────────────────┴────────────────────┐        │
│                          v                                         v        │
│         ┌───────────────────────────┐         ┌──────────────────────────┐  │
│         │ classify_gesture_rules    │         │   ml_predict_baseline    │  │
│         │ (threshold-based)         │         │   (trained model)        │  │
│         └───────────────────────────┘         └──────────────────────────┘  │
│                          │                                         │        │
│                          └─────────────────────┬────────────────────┘        │
│                                               v                             │
│                              ┌─────────────────────────────────────────┐    │
│                              │         plot_diagnostics               │    │
│                              │  - orientation, segmentation overlay   │    │
│                              │  - covariance traces, innovations      │    │
│                              └─────────────────────────────────────────┘    │
│                                               │                             │
│                                               v                             │
│                              ┌─────────────────────────────────────────┐    │
│                              │         RESULT: Gesture Label          │    │
│                              │         + Confidence Score             │    │
│                              └─────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Specification

### Sensors Expected

| Sensor | Units | Typical Range | Notes |
|--------|-------|---------------|-------|
| Accelerometer | m/s² | ±20 m/s² (2g) | Includes gravity when stationary |
| Gyroscope | rad/s | ±8 rad/s | May need °/s → rad/s conversion |
| Magnetometer | µT | ±100 µT | Heavily disturbed indoors |
| Device Attitude | quaternion | [-1,1] | Reference only if available |

### Sampling & Timestamps

- Expected Fs: 50–100 Hz from MATLAB Mobile
- Handle variable dt (compute from timestamps)
- Option to resample to fixed rate (e.g., 100 Hz)
- Timestamp in seconds from start of recording

### Coordinate Frames

```
Phone/Body Frame (ENU convention adopted):
        +Z (out of screen)
         │
         │    +Y (top of phone)
         │   /
         │  /
         │ /
         └──────── +X (right side of phone)

World Frame (NED or ENU - configurable):
  - Gravity: -Z in ENU (down)
  - Magnetic North: +X or +Y component
```

---

## 7. Algorithms / Technical Approach

### 7.1 Preprocessing

1. **Unit Conversion**: Ensure acc in m/s², gyro in rad/s, mag in µT
2. **Axis Alignment**: Verify right-hand rule, consistent across sensors
3. **Bias Estimation**: Static segment average for gyro bias; accelerometer gravity subtraction
4. **Low-pass Filtering**: Optional 20Hz cutoff for noise reduction
5. **Magnetometer Calibration**: Hard-iron offset via sphere fitting (simplified)

### 7.2 Quaternion Mathematics

**Convention**: Hamilton quaternion q = [w, x, y, z] where w is scalar part.

**Key Operations**:
```
Multiplication: q1 ⊗ q2 (Hamilton product)
Rotation: v' = q ⊗ [0,v] ⊗ q*
From angular velocity: q_delta = exp(ω*dt/2) ≈ [1, ω*dt/2]
Normalization: q = q / ||q|| (after every update!)
```

**Why Quaternions over Euler Angles?**
- No gimbal lock (singularity at ±90° pitch)
- Smooth interpolation (SLERP)
- Efficient composition (single quaternion multiply vs 3 rotation matrices)
- Constant 4 parameters vs variable singularities

### 7.3 Extended Kalman Filter (EKF)

**State Vector** (7 states):
```
x = [q_w, q_x, q_y, q_z, b_gx, b_gy, b_gz]^T
```
- q: attitude quaternion (4 components, constrained to unit norm)
- b_g: gyroscope bias (3 components, modeled as random walk)

**Process Model**:
```
q_{k+1} = q_k ⊗ q_delta(ω_m - b_g, dt)
b_{k+1} = b_k + w_b  (random walk)

where q_delta(ω,dt) ≈ [cos(|ω|dt/2), ω/|ω| * sin(|ω|dt/2)]
```

**Measurement Models**:
1. **Gravity** (from accelerometer when |a| ≈ g):
   - Predicted: g_body = R(q)^T * g_world
   - Innovation: z_g = a_measured - g_body
   
2. **Magnetic field** (from magnetometer):
   - Predicted: m_body = R(q)^T * m_world
   - Innovation: z_m = m_measured - m_body

**Jacobians**: Computed analytically for rotation matrix elements.

**Normalization**: After every state update, re-normalize quaternion.

### 7.4 Linear Kalman Filter (KF)

**Purpose**: Estimate linear velocity and position from world-frame acceleration.

**State Vector** (6 states):
```
x = [v_x, v_y, v_z, p_x, p_y, p_z]^T
```

**Process Model** (constant acceleration):
```
v_{k+1} = v_k + a_world * dt
p_{k+1} = p_k + v_k * dt + 0.5 * a_world * dt²
```

**Measurements**:
- **Zero-velocity update (ZUPT)**: When stationary detected, measure v = 0
- This prevents unbounded drift in position

**Why Linear KF Here?**
- After removing gravity and rotating to world frame, the dynamics are linear
- Demonstrates KF vs EKF distinction (EKF needed for nonlinear attitude)

### 7.5 Gesture Classification

**Rule-Based Approach**:
1. Extract features: peak gyro, dominant axis, duration, energy
2. Apply threshold trees:
   - High gyro_z + short duration → "Twist"
   - High gyro_x + positive → "Flip Up"
   - Oscillating gyro + multiple zero-crossings → "Shake"

**ML Baseline** (for comparison):
- Feature vector → kNN or SVM
- 5-fold cross-validation
- Confusion matrix output

---

## 8. Gesture Set Definition

| Gesture | Duration | Primary Sensor Signature | Description |
|---------|----------|-------------------------|-------------|
| **Flip Up** | 0.5-1.0s | High +gyro_x, acc_z changes sign | Rotate phone 90° towards user |
| **Flip Down** | 0.5-1.0s | High -gyro_x, acc_z changes sign | Rotate phone 90° away from user |
| **Shake** | 0.5-1.5s | Oscillating gyro_y/z, high acc RMS | Rapid left-right shaking |
| **Twist** | 0.3-0.8s | High gyro_z peak, acc stable | Rotate phone about screen axis |
| **Push Forward** | 0.3-0.7s | High +acc_y spike, low rotation | Thrust phone forward |
| **Circle** | 1.0-2.0s | Sinusoidal gyro_x and gyro_y | Circular motion in front |

---

## 9. ML Baseline Path

### Data Format for Training
```matlab
% gestures_labeled.mat contains:
%   labels:   Nx1 cell array of gesture names
%   features: NxM matrix of extracted features
%   featureNames: 1xM cell array of feature descriptions
```

### Training Pipeline
1. Load labeled examples
2. Extract features via `extract_features.m`
3. Train kNN/SVM using `fitcknn` or `fitcsvm`
4. Evaluate via cross-validation
5. Save model to `/models/gesture_model.mat`

### Extension Path
- Replace kNN with LSTM for sequence modeling
- Use Deep Learning Toolbox `trainNetwork`
- Time-series classification via `sequenceInputLayer`

---

## 10. Validation & Metrics

### Quantitative Metrics
- **Classification Accuracy**: % correct on test set
- **Quaternion Norm**: Should be 1.0 ± 0.001 at all times
- **Gravity Alignment Error**: |g_estimated - g_actual| < 0.5 m/s²
- **Yaw Drift**: < 5°/minute during static periods

### Sanity Check Mode
Run `test_ekf_static.m`:
- Feed 10 seconds of stationary data
- Verify orientation converges to stable estimate
- Verify quaternion norm maintained
- Plot covariance trace (should decrease then plateau)

---

## 11. Risks / Failure Modes

| Risk | Impact | Mitigation |
|------|--------|------------|
| Magnetometer interference | Yaw drift, false headings | Mag outlier rejection, rely more on gyro |
| Variable sampling rate | Integration errors | Resample to fixed rate |
| User variability | Poor classification | More training data, robust features |
| Phone orientation change | Wrong gesture mapping | Calibration prompt, reference pose |

---

## 12. Acceptance Criteria

- [ ] Run `main_gesture_demo.m` and receive a gesture label from sample log
- [ ] Plots show: raw IMU, fused orientation (Euler for viz), segmentation overlay
- [ ] EKF quaternion norm stays within [0.999, 1.001]
- [ ] Linear KF produces velocity estimate that ZUPT resets to near-zero
- [ ] Rule-based classifier outputs label with confidence
- [ ] README references MathWorks documentation for each technique

---

# PART B: QUICK START GUIDE

## Installation

1. Clone or copy the `gesture-sa-sensor-fusion/` folder
2. Open MATLAB and navigate to the project root
3. Add source to path:
```matlab
addpath(genpath('src'));
```

## Data Export from MATLAB Mobile

1. Install **MATLAB Mobile** from App Store
2. Open MATLAB Mobile → **Sensors** tab
3. Enable: Accelerometer, Gyroscope, Magnetometer (and Orientation if available)
4. Set sample rate to 100 Hz if possible
5. Press **Start** → perform gestures → **Stop**
6. **Send to MATLAB Drive** or **Email** as MAT/CSV
7. Place file in `data/raw/`

## Running the Demo

```matlab
% From project root
main_gesture_demo
```

Expected output:
- Diagnostic figures showing raw and filtered IMU
- Orientation estimation plots
- Gesture segmentation visualization
- Console output with detected gesture label

## Generating Synthetic Test Data

```matlab
% Create fake gesture data for pipeline testing
[data] = generate_synthetic_gesture('twist', 100);
save('data/raw/synthetic_twist.mat', 'data');
```

---

# PART C: TECHNICAL DOCUMENTATION

## Why Quaternions vs Euler Angles?

**Euler angles** (roll, pitch, yaw) suffer from:
- **Gimbal lock**: When pitch = ±90°, roll and yaw become indistinguishable
- **Order dependence**: ZYX vs XYZ gives different results
- **Discontinuities**: 359° → 0° wrap-around causes issues

**Quaternions** provide:
- **Singularity-free** representation
- **Smooth interpolation** (SLERP)
- **Efficient composition** (single multiplication)
- **Minimal parameters** (4 vs 9 for rotation matrix)

## Why EKF for Attitude?

The attitude dynamics are **nonlinear**:
- Quaternion composition: q_new = q_old ⊗ q_delta
- Measurement prediction: g_body = R(q)^T * g_world

Standard KF assumes linear state transition and measurement. EKF linearizes around current estimate via Jacobians, enabling optimal filtering for nonlinear systems.

## Why Linear KF for Motion?

After removing gravity and transforming to world frame:
```
v_{k+1} = v_k + a_world * dt  (linear!)
p_{k+1} = p_k + v_k * dt       (linear!)
```

This is a textbook linear state-space model. Using EKF here would be unnecessary and computationally wasteful. The linear KF demonstrates the classical algorithm where appropriate.

## MathWorks Documentation References

| Topic | Search Term | URL |
|-------|-------------|-----|
| Quaternion class | "quaternion" | https://www.mathworks.com/help/fusion/ref/quaternion.html |
| IMU sensor fusion | "imufilter" | https://www.mathworks.com/help/fusion/ref/imufilter-system-object.html |
| AHRS filter | "ahrsfilter" | https://www.mathworks.com/help/fusion/ref/ahrsfilter-system-object.html |
| Complementary filter | "complementaryFilter" | https://www.mathworks.com/help/fusion/ref/complementaryfilter-system-object.html |
| Extended Kalman Filter | "extendedKalmanFilter" | https://www.mathworks.com/help/control/ref/extendedkalmanfilter.html |
| Tracking EKF | "trackingEKF" | https://www.mathworks.com/help/fusion/ref/trackingekf.html |
| Table import | "readtable" | https://www.mathworks.com/help/matlab/ref/readtable.html |
| Timetable | "timetable" | https://www.mathworks.com/help/matlab/ref/timetable.html |
| kNN classifier | "fitcknn" | https://www.mathworks.com/help/stats/fitcknn.html |
| SVM classifier | "fitcsvm" | https://www.mathworks.com/help/stats/fitcsvm.html |

---

## Troubleshooting

### Magnetometer Disturbance Symptoms
- Yaw estimate drifts or jumps suddenly
- Heading doesn't match physical orientation
- **Solution**: Increase mag measurement covariance (R_mag), enable outlier rejection

### Axis Mismatch Symptoms
- Rotation about one axis shows up on another
- Gravity points wrong direction
- **Solution**: Check coordinate frame conventions in `config_params.m`

### Variable dt Symptoms
- Jerky orientation estimates
- Velocity integration goes wild
- **Solution**: Use `preprocess_imu.m` resampling, or compute dt per sample

### Quaternion Drift
- Norm slowly deviates from 1.0
- Estimates become unstable
- **Solution**: Normalize after every EKF update (built into our code)

---

## Casio fx-991EX Sanity Checks

For students wanting to verify calculations by hand:

### 1. Quaternion Normalization
```
Given q = [0.707, 0.707, 0, 0]
||q||² = 0.707² + 0.707² + 0² + 0² = 0.5 + 0.5 = 1.0 ✓
```

### 2. RMS Calculation
```
For accelerometer segment: [9.8, 10.2, 9.6, 10.0, 9.9]
Mean = 9.9
RMS = √[(9.8² + 10.2² + 9.6² + 10.0² + 9.9²)/5]
    = √[(96.04 + 104.04 + 92.16 + 100 + 98.01)/5]
    = √[490.25/5] = √98.05 = 9.90
```

### 3. 2x2 Kalman Gain Intuition
```
P = [1 0; 0 1]  (initial covariance)
H = [1 0]       (measure first state)
R = 0.1         (measurement noise)

S = H*P*H' + R = 1*1*1 + 0.1 = 1.1
K = P*H'/S = [1 0; 0 1]*[1;0]/1.1 = [0.91; 0]

Interpretation: 91% weight on measurement for first state
```

---

## Adding New Gestures

1. **Define in config_params.m**:
```matlab
params.gestures.labels = {'flip_up','flip_down','shake','twist','push','circle','NEW_GESTURE'};
```

2. **Add signature in classify_gesture_rules.m**:
```matlab
if feat.peak_gyro_y > params.gestures.NEW_THRESHOLD
    cls.label = 'NEW_GESTURE';
    cls.score = 0.8;
    cls.reason = 'High Y rotation detected';
    return;
end
```

3. **Collect labeled examples** and retrain ML model.

---

## How to Train ML Model

1. **Collect data**: Record multiple examples of each gesture
2. **Label**: Create CSV with columns: filename, label
3. **Run training**:
```matlab
ml_train_baseline('data/labeled/gesture_labels.csv');
```
4. **Evaluate**: Check confusion matrix in console output
5. **Use**: Model saved to `models/gesture_model.mat`

---

## File Structure Reference

```
gesture-sa-sensor-fusion/
├── README.md                    # This file
├── data/
│   ├── raw/                     # Raw sensor logs
│   └── labeled/                 # Labeled gesture data
├── outputs/
│   ├── figures/                 # Generated plots
│   └── logs/                    # Run logs
├── models/                      # Trained ML models
├── src/
│   ├── main/
│   │   ├── main_gesture_demo.m  # Main entry point
│   │   ├── run_training.m       # ML training script
│   │   └── run_batch_eval.m     # Batch evaluation
│   ├── io/
│   │   ├── read_phone_data.m    # Data import
│   │   └── export_helpers.m     # Export utilities
│   ├── preprocess/
│   │   ├── preprocess_imu.m     # IMU preprocessing
│   │   ├── calibrate_mag_simple.m
│   │   └── resample_signals.m
│   ├── fusion/
│   │   ├── ekf_attitude_quat.m  # EKF implementation
│   │   ├── kf_linear_motion.m   # Linear KF
│   │   └── complementary_filter.m
│   ├── gestures/
│   │   ├── segment_gesture.m    # Gesture segmentation
│   │   ├── extract_features.m   # Feature extraction
│   │   └── classify_gesture_rules.m
│   ├── ml/
│   │   ├── ml_train_baseline.m  # ML training
│   │   └── ml_predict_baseline.m
│   ├── viz/
│   │   ├── plot_diagnostics.m   # Plotting functions
│   │   └── plot_gesture_segment.m
│   └── utils/
│       ├── config_params.m      # Configuration
│       ├── quat_utils.m         # Quaternion utilities
│       ├── time_utils.m
│       └── assert_utils.m
└── tests/
    ├── test_quat_math.m
    ├── test_import.m
    └── test_ekf_static.m
```

---

## License

This project is provided for educational purposes.

## Author

Rashaan Palmer
