function params = config_params()
%CONFIG_PARAMS Central configuration for all tuning parameters
%   params = config_params() returns a struct containing all tunable
%   parameters for the gesture recognition pipeline.
%
%   USAGE:
%       params = config_params();
%       params.ekf.Q_gyro = 1e-5;  % Override if needed
%
%   STRUCTURE:
%       params.sampling     - Sampling rate assumptions
%       params.preprocess   - Preprocessing parameters
%       params.ekf          - Extended Kalman Filter tuning
%       params.kf           - Linear Kalman Filter tuning
%       params.segmentation - Gesture segmentation thresholds
%       params.features     - Feature extraction settings
%       params.gestures     - Gesture definitions and thresholds
%       params.ml           - Machine learning settings
%       params.viz          - Visualization options
%       params.paths        - Default file paths
%
%   Author: Sensor Fusion Demo
%   Date: 2024

    %% ==================== SAMPLING ====================
    params.sampling.expected_Fs = 100;      % Hz, expected sample rate
    params.sampling.min_Fs = 20;            % Hz, minimum acceptable
    params.sampling.max_Fs = 200;           % Hz, maximum acceptable
    params.sampling.resample_target = 100;  % Hz, resample to this rate
    params.sampling.do_resample = true;     % Enable resampling
    
    %% ==================== PREPROCESSING ====================
    % Unit conversions (set to 1 if already correct)
    params.preprocess.acc_scale = 1;        % Multiply acc by this (e.g., 9.81 if in g)
    params.preprocess.gyr_scale = 1;        % Multiply gyro by this (e.g., pi/180 if in deg/s)
    params.preprocess.mag_scale = 1;        % Multiply mag by this
    
    % Filtering
    params.preprocess.lpf_cutoff = 25;      % Hz, low-pass filter cutoff
    params.preprocess.lpf_order = 2;        % Butterworth filter order
    params.preprocess.use_lpf = true;       % Enable low-pass filtering
    
    % Bias estimation (from static segments)
    params.preprocess.static_threshold = 0.5;   % rad/s, gyro magnitude threshold for static
    params.preprocess.static_window = 50;       % samples, window for static detection
    params.preprocess.estimate_gyro_bias = true;
    
    % Magnetometer calibration
    params.preprocess.mag_calibration = true;
    params.preprocess.mag_outlier_threshold = 3;  % sigma for outlier rejection
    
    %% ==================== COORDINATE FRAMES ====================
    % World frame gravity direction (in world coordinates)
    params.frames.gravity_world = [0; 0; -9.81];  % ENU: gravity is -Z
    
    % Reference magnetic field direction (normalized, approximate)
    % This should be adjusted for your location
    % For simplicity, we assume magnetic north aligns with +X (east-ish)
    params.frames.mag_ref_world = [1; 0; 0];  % Normalized reference
    
    % Body frame convention: standard phone orientation
    % +X = right, +Y = up (top of phone), +Z = out of screen
    
    %% ==================== EKF ATTITUDE ====================
    % State: [q_w, q_x, q_y, q_z, b_gx, b_gy, b_gz]
    
    % Process noise covariance Q (7x7 diagonal approximation)
    params.ekf.Q_quat = 1e-6;       % Quaternion process noise
    params.ekf.Q_gyro_bias = 1e-8;  % Gyro bias random walk
    
    % Measurement noise covariance R
    params.ekf.R_acc = 0.5;         % Accelerometer measurement noise (m/s²)²
    params.ekf.R_mag = 2.0;         % Magnetometer measurement noise (µT)²
    
    % Initial state covariance P0
    params.ekf.P0_quat = 0.1;       % Initial quaternion uncertainty
    params.ekf.P0_bias = 0.01;      % Initial bias uncertainty (rad/s)²
    
    % Initial gyro bias estimate
    params.ekf.init_gyro_bias = [0; 0; 0];
    
    % Adaptive tuning
    params.ekf.acc_magnitude_window = [8, 12];  % Accept acc update if |a| in this range (m/s²)
    params.ekf.use_mag_update = true;           % Use magnetometer in updates
    params.ekf.mag_rejection_threshold = 5;     % Reject mag if innovation > this (sigma)
    
    %% ==================== LINEAR KF (Velocity/Position) ====================
    % State: [v_x, v_y, v_z, p_x, p_y, p_z]
    
    % Process noise
    params.kf.Q_velocity = 0.1;     % Velocity process noise (m/s)²
    params.kf.Q_position = 0.01;    % Position process noise (m)²
    
    % Measurement noise
    params.kf.R_zupt = 0.01;        % Zero-velocity update noise (m/s)²
    
    % Initial covariance
    params.kf.P0_velocity = 1;      % Initial velocity uncertainty (m/s)²
    params.kf.P0_position = 0.1;    % Initial position uncertainty (m)²
    
    % ZUPT detection
    params.kf.zupt_gyro_threshold = 0.2;   % rad/s, gyro threshold for stationary
    params.kf.zupt_acc_threshold = 1.0;    % m/s², acc variance threshold
    params.kf.zupt_window = 10;            % samples
    
    %% ==================== COMPLEMENTARY FILTER ====================
    params.comp.alpha = 0.98;       % Gyro weight (0.95-0.99 typical)
    params.comp.beta = 0.1;         % Magnetometer correction gain
    
    %% ==================== GESTURE SEGMENTATION ====================
    % Energy-based segmentation
    params.segmentation.method = 'energy';  % 'energy' or 'peaks'
    
    % Energy thresholds
    params.segmentation.energy_low = 0.5;   % rad/s, below this = quiet
    params.segmentation.energy_high = 1.5;  % rad/s, above this = active
    
    % Hysteresis parameters
    params.segmentation.min_duration = 0.2;   % seconds, minimum gesture length
    params.segmentation.max_duration = 3.0;   % seconds, maximum gesture length
    params.segmentation.pre_buffer = 0.1;     % seconds, include before trigger
    params.segmentation.post_buffer = 0.1;    % seconds, include after end
    
    % Multiple gesture handling
    params.segmentation.max_gestures = 5;     % Maximum gestures to detect
    params.segmentation.min_gap = 0.3;        % seconds, minimum between gestures
    
    %% ==================== FEATURE EXTRACTION ====================
    params.features.compute_fft = true;       % Compute frequency features
    params.features.fft_nfft = 256;           % FFT size
    params.features.freq_bands = [0.5, 5; 5, 15; 15, 25];  % Hz, frequency bands
    
    % Feature list to compute
    params.features.list = {
        'duration',           % Gesture duration (s)
        'rms_acc_x', 'rms_acc_y', 'rms_acc_z',  % RMS acceleration per axis
        'rms_gyr_x', 'rms_gyr_y', 'rms_gyr_z',  % RMS gyro per axis
        'peak_gyr_x', 'peak_gyr_y', 'peak_gyr_z',  % Peak gyro per axis
        'peak_gyr_axis',      % Dominant rotation axis (1,2,3 for x,y,z)
        'total_rotation',     % Total rotation angle (rad)
        'zero_cross_gyr',     % Zero-crossing count in gyro
        'energy_ratio_gyr',   % Energy ratio between axes
        'acc_range',          % Max - min acceleration magnitude
    };
    
    %% ==================== GESTURE DEFINITIONS ====================
    params.gestures.labels = {
        'flip_up',      % Rotate phone towards user (positive gyro_x)
        'flip_down',    % Rotate phone away from user (negative gyro_x)
        'shake',        % Rapid left-right shaking
        'twist',        % Rotate about screen axis (gyro_z)
        'push_forward', % Thrust forward (high acc_y)
        'circle',       % Circular motion
        'unknown'       % Catch-all
    };
    
    % Rule-based thresholds
    params.gestures.rules.flip_gyro_threshold = 2.5;    % rad/s
    params.gestures.rules.shake_oscillation_count = 3;  % zero crossings
    params.gestures.rules.shake_gyro_threshold = 3.0;   % rad/s
    params.gestures.rules.twist_gyro_z_threshold = 3.0; % rad/s
    params.gestures.rules.push_acc_threshold = 5.0;     % m/s² above gravity
    params.gestures.rules.circle_duration_min = 0.8;    % seconds
    params.gestures.rules.circle_phase_threshold = 0.5; % rad, phase offset
    
    %% ==================== MACHINE LEARNING ====================
    params.ml.method = 'knn';           % 'knn', 'svm', 'tree'
    params.ml.knn_k = 5;                % Number of neighbors
    params.ml.svm_kernel = 'rbf';       % SVM kernel type
    params.ml.cross_val_folds = 5;      % Cross-validation folds
    params.ml.normalize_features = true; % Z-score normalization
    params.ml.model_path = 'models/gesture_model.mat';
    
    %% ==================== VISUALIZATION ====================
    params.viz.show_plots = true;       % Display plots
    params.viz.save_plots = false;      % Save plots to file
    params.viz.figure_format = 'png';   % 'png', 'fig', 'pdf'
    params.viz.dpi = 150;               % Resolution
    
    % Plot colors
    params.viz.colors.acc = [0.8, 0.2, 0.2];    % Red for acceleration
    params.viz.colors.gyr = [0.2, 0.2, 0.8];    % Blue for gyroscope
    params.viz.colors.mag = [0.2, 0.8, 0.2];    % Green for magnetometer
    params.viz.colors.est = [0.8, 0.5, 0.0];    % Orange for estimates
    params.viz.colors.segment = [0.9, 0.9, 0.5]; % Yellow for segments
    
    %% ==================== FILE PATHS ====================
    params.paths.data_raw = 'data/raw/';
    params.paths.data_labeled = 'data/labeled/';
    params.paths.outputs = 'outputs/';
    params.paths.figures = 'outputs/figures/';
    params.paths.logs = 'outputs/logs/';
    params.paths.models = 'models/';
    
    %% ==================== PHYSICAL CONSTANTS ====================
    params.constants.g = 9.81;          % m/s², gravity magnitude
    params.constants.deg2rad = pi/180;  % Conversion factor
    params.constants.rad2deg = 180/pi;  % Conversion factor

end
