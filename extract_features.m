function feat = extract_features(imu, seg, est, params)
%EXTRACT_FEATURES Extract features from a segmented gesture
%   feat = extract_features(imu, seg, est, params) computes a feature vector
%   from the primary gesture segment.
%
%   INPUTS:
%       imu     - Preprocessed IMU data
%       seg     - Segmentation results from segment_gesture()
%       est     - Attitude estimation from ekf_attitude_quat() (optional)
%       params  - Configuration from config_params() (optional)
%
%   OUTPUT:
%       feat - Feature struct:
%           .x          - 1xM feature vector (numeric)
%           .names      - 1xM cell array of feature names
%           .values     - Struct with named feature values
%           .debug      - Additional debug information
%
%   FEATURES COMPUTED:
%       - Duration (seconds)
%       - RMS acceleration (per axis)
%       - RMS gyroscope (per axis)
%       - Peak gyroscope (per axis)
%       - Dominant rotation axis
%       - Total rotation angle
%       - Zero-crossing counts
%       - Energy distribution
%       - Frequency features (optional)
%
%   EXAMPLE:
%       imu = preprocess_imu(data, params);
%       seg = segment_gesture(imu, params);
%       est = ekf_attitude_quat(imu, params);
%       feat = extract_features(imu, seg, est, params);
%       
%       disp(feat.names);
%       disp(feat.x);
%
%   Author: Sensor Fusion Demo
%   Date: 2024

    %% Default parameters
    if nargin < 4
        params = config_params();
    end
    if nargin < 3
        est = [];
    end
    
    %% Get gesture window
    if seg.n_gestures > 0
        idx_start = seg.winIdx(1);
        idx_end = seg.winIdx(2);
    else
        idx_start = 1;
        idx_end = length(imu.t);
    end
    
    % Extract windowed data
    t_win = imu.t(idx_start:idx_end);
    acc_win = imu.acc(idx_start:idx_end, :);
    gyr_win = imu.gyr(idx_start:idx_end, :);
    
    n_samples = length(t_win);
    Fs = imu.Fs;
    
    %% Initialize outputs
    feat.values = struct();
    feat.names = {};
    feat.x = [];
    
    %% Duration
    feat.values.duration = t_win(end) - t_win(1);
    add_feature('duration', feat.values.duration);
    
    %% RMS Acceleration (per axis)
    feat.values.rms_acc_x = rms(acc_win(:,1));
    feat.values.rms_acc_y = rms(acc_win(:,2));
    feat.values.rms_acc_z = rms(acc_win(:,3));
    feat.values.rms_acc_total = rms(sqrt(sum(acc_win.^2, 2)));
    
    add_feature('rms_acc_x', feat.values.rms_acc_x);
    add_feature('rms_acc_y', feat.values.rms_acc_y);
    add_feature('rms_acc_z', feat.values.rms_acc_z);
    add_feature('rms_acc_total', feat.values.rms_acc_total);
    
    %% RMS Gyroscope (per axis)
    feat.values.rms_gyr_x = rms(gyr_win(:,1));
    feat.values.rms_gyr_y = rms(gyr_win(:,2));
    feat.values.rms_gyr_z = rms(gyr_win(:,3));
    feat.values.rms_gyr_total = rms(sqrt(sum(gyr_win.^2, 2)));
    
    add_feature('rms_gyr_x', feat.values.rms_gyr_x);
    add_feature('rms_gyr_y', feat.values.rms_gyr_y);
    add_feature('rms_gyr_z', feat.values.rms_gyr_z);
    add_feature('rms_gyr_total', feat.values.rms_gyr_total);
    
    %% Peak Gyroscope (per axis, signed)
    [~, peak_idx_x] = max(abs(gyr_win(:,1)));
    [~, peak_idx_y] = max(abs(gyr_win(:,2)));
    [~, peak_idx_z] = max(abs(gyr_win(:,3)));
    
    feat.values.peak_gyr_x = gyr_win(peak_idx_x, 1);
    feat.values.peak_gyr_y = gyr_win(peak_idx_y, 2);
    feat.values.peak_gyr_z = gyr_win(peak_idx_z, 3);
    
    add_feature('peak_gyr_x', feat.values.peak_gyr_x);
    add_feature('peak_gyr_y', feat.values.peak_gyr_y);
    add_feature('peak_gyr_z', feat.values.peak_gyr_z);
    
    %% Peak Gyroscope magnitudes (absolute)
    feat.values.peak_gyr_x_abs = abs(feat.values.peak_gyr_x);
    feat.values.peak_gyr_y_abs = abs(feat.values.peak_gyr_y);
    feat.values.peak_gyr_z_abs = abs(feat.values.peak_gyr_z);
    
    add_feature('peak_gyr_x_abs', feat.values.peak_gyr_x_abs);
    add_feature('peak_gyr_y_abs', feat.values.peak_gyr_y_abs);
    add_feature('peak_gyr_z_abs', feat.values.peak_gyr_z_abs);
    
    %% Dominant rotation axis
    peak_abs = [feat.values.peak_gyr_x_abs, feat.values.peak_gyr_y_abs, feat.values.peak_gyr_z_abs];
    [~, dominant_axis] = max(peak_abs);
    feat.values.dominant_axis = dominant_axis;  % 1=X, 2=Y, 3=Z
    add_feature('dominant_axis', dominant_axis);
    
    %% Total rotation angle (integrate gyro magnitude)
    gyr_mag = sqrt(sum(gyr_win.^2, 2));
    dt = 1 / Fs;
    feat.values.total_rotation = sum(gyr_mag) * dt;  % radians
    add_feature('total_rotation', feat.values.total_rotation);
    
    %% Zero-crossing counts (indicates oscillation)
    zc_x = count_zero_crossings(gyr_win(:,1));
    zc_y = count_zero_crossings(gyr_win(:,2));
    zc_z = count_zero_crossings(gyr_win(:,3));
    
    feat.values.zero_cross_gyr_x = zc_x;
    feat.values.zero_cross_gyr_y = zc_y;
    feat.values.zero_cross_gyr_z = zc_z;
    feat.values.zero_cross_gyr_total = zc_x + zc_y + zc_z;
    
    add_feature('zero_cross_gyr_x', zc_x);
    add_feature('zero_cross_gyr_y', zc_y);
    add_feature('zero_cross_gyr_z', zc_z);
    add_feature('zero_cross_gyr_total', zc_x + zc_y + zc_z);
    
    %% Energy distribution ratios
    energy_x = sum(gyr_win(:,1).^2);
    energy_y = sum(gyr_win(:,2).^2);
    energy_z = sum(gyr_win(:,3).^2);
    energy_total = energy_x + energy_y + energy_z + 1e-10;  % Avoid division by zero
    
    feat.values.energy_ratio_x = energy_x / energy_total;
    feat.values.energy_ratio_y = energy_y / energy_total;
    feat.values.energy_ratio_z = energy_z / energy_total;
    
    add_feature('energy_ratio_x', feat.values.energy_ratio_x);
    add_feature('energy_ratio_y', feat.values.energy_ratio_y);
    add_feature('energy_ratio_z', feat.values.energy_ratio_z);
    
    %% Acceleration range (max - min magnitude)
    acc_mag = sqrt(sum(acc_win.^2, 2));
    feat.values.acc_range = max(acc_mag) - min(acc_mag);
    add_feature('acc_range', feat.values.acc_range);
    
    %% Mean and variance
    feat.values.mean_gyr_x = mean(gyr_win(:,1));
    feat.values.mean_gyr_y = mean(gyr_win(:,2));
    feat.values.mean_gyr_z = mean(gyr_win(:,3));
    feat.values.var_gyr_total = var(gyr_mag);
    
    add_feature('mean_gyr_x', feat.values.mean_gyr_x);
    add_feature('mean_gyr_y', feat.values.mean_gyr_y);
    add_feature('mean_gyr_z', feat.values.mean_gyr_z);
    add_feature('var_gyr_total', feat.values.var_gyr_total);
    
    %% Phase features (for circular motion detection)
    % Check if X and Y gyro are 90° out of phase
    if n_samples > 10
        % Normalize signals
        gx_norm = (gyr_win(:,1) - mean(gyr_win(:,1))) / (std(gyr_win(:,1)) + 1e-10);
        gy_norm = (gyr_win(:,2) - mean(gyr_win(:,2))) / (std(gyr_win(:,2)) + 1e-10);
        
        % Cross-correlation at lag 0 and peak
        [xcorr_val, lags] = xcorr(gx_norm, gy_norm, round(n_samples/4));
        [~, max_idx] = max(abs(xcorr_val));
        phase_lag = lags(max_idx);
        
        feat.values.phase_lag_xy = phase_lag / Fs;  % In seconds
        add_feature('phase_lag_xy', feat.values.phase_lag_xy);
    else
        feat.values.phase_lag_xy = 0;
        add_feature('phase_lag_xy', 0);
    end
    
    %% Frequency features (optional)
    if params.features.compute_fft && n_samples >= 16
        nfft = min(params.features.fft_nfft, 2^nextpow2(n_samples));
        
        % FFT of gyro magnitude
        Y = fft(gyr_mag - mean(gyr_mag), nfft);
        P = abs(Y(1:nfft/2+1)).^2;
        f = Fs * (0:(nfft/2)) / nfft;
        
        % Dominant frequency
        [~, dom_idx] = max(P(2:end));  % Exclude DC
        feat.values.dominant_freq = f(dom_idx + 1);
        add_feature('dominant_freq', feat.values.dominant_freq);
        
        % Energy in frequency bands
        bands = params.features.freq_bands;
        for b = 1:size(bands, 1)
            f_low = bands(b, 1);
            f_high = bands(b, 2);
            band_idx = f >= f_low & f <= f_high;
            band_energy = sum(P(band_idx)) / (sum(P) + 1e-10);
            
            feat_name = sprintf('freq_band_%d_%d', round(f_low), round(f_high));
            feat.values.(matlab.lang.makeValidName(feat_name)) = band_energy;
            add_feature(feat_name, band_energy);
        end
    end
    
    %% Orientation change (if attitude estimation available)
    if ~isempty(est) && isfield(est, 'euler')
        euler_win = est.euler(idx_start:idx_end, :);
        
        % Total Euler angle change
        euler_change = euler_win(end,:) - euler_win(1,:);
        
        % Wrap to [-pi, pi]
        euler_change = wrapToPi(euler_change);
        
        feat.values.delta_roll = euler_change(1);
        feat.values.delta_pitch = euler_change(2);
        feat.values.delta_yaw = euler_change(3);
        
        add_feature('delta_roll', euler_change(1));
        add_feature('delta_pitch', euler_change(2));
        add_feature('delta_yaw', euler_change(3));
    end
    
    %% Debug info
    feat.debug.n_samples = n_samples;
    feat.debug.idx_start = idx_start;
    feat.debug.idx_end = idx_end;
    feat.debug.t_start = t_win(1);
    feat.debug.t_end = t_win(end);
    
    %% Nested function to add features consistently
    function add_feature(name, value)
        feat.names{end+1} = name;
        feat.x(end+1) = value;
    end
    
end

%% ==================== HELPER FUNCTIONS ====================

function count = count_zero_crossings(signal)
%COUNT_ZERO_CROSSINGS Count number of zero crossings in signal
    
    % Remove mean to handle offset signals
    signal = signal - mean(signal);
    
    % Count sign changes
    signs = sign(signal);
    signs(signs == 0) = 1;  % Treat exactly zero as positive
    count = sum(abs(diff(signs)) == 2);
end

function angle = wrapToPi(angle)
%WRAPTOPI Wrap angle to [-pi, pi]
    angle = mod(angle + pi, 2*pi) - pi;
end
