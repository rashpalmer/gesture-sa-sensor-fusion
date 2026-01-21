function seg = segment_gesture(imu, params)
%SEGMENT_GESTURE Detect and segment gestures from IMU data
%   seg = segment_gesture(imu, params) identifies gesture boundaries using
%   energy-based thresholding with hysteresis.
%
%   INPUTS:
%       imu     - Preprocessed IMU data from preprocess_imu()
%       params  - Configuration from config_params() (optional)
%
%   OUTPUT:
%       seg - Segmentation results struct:
%           .windows    - Mx2 matrix of [start_idx, end_idx] for each gesture
%           .n_gestures - Number of gestures detected
%           .energy     - Nx1 motion energy signal
%           .state      - Nx1 state machine output (0=quiet, 1=active)
%           .primary    - Index of most prominent gesture
%           .winIdx     - [start, end] of primary gesture
%           .score      - Confidence score for primary gesture
%
%   ALGORITHM:
%       1. Compute motion energy from gyroscope magnitude
%       2. Apply hysteresis thresholding (high/low thresholds)
%       3. Find contiguous active regions
%       4. Filter by duration constraints
%       5. Rank by total energy, select primary gesture
%
%   EXAMPLE:
%       imu = preprocess_imu(data, params);
%       seg = segment_gesture(imu, params);
%       
%       % Plot with segmentation overlay
%       plot(imu.t, imu.gyr);
%       hold on;
%       for i = 1:seg.n_gestures
%           xline(imu.t(seg.windows(i,1)), 'g', 'LineWidth', 2);
%           xline(imu.t(seg.windows(i,2)), 'r', 'LineWidth', 2);
%       end
%
%   Author: Sensor Fusion Demo
%   Date: 2024

    %% Default parameters
    if nargin < 2
        params = config_params();
    end
    
    fprintf('Segmenting gestures...\n');
    
    %% Initialize
    n = length(imu.t);
    dt = mean(imu.dt);
    Fs = imu.Fs;
    
    % Get thresholds from params
    energy_low = params.segmentation.energy_low;
    energy_high = params.segmentation.energy_high;
    min_duration = params.segmentation.min_duration;
    max_duration = params.segmentation.max_duration;
    pre_buffer = params.segmentation.pre_buffer;
    post_buffer = params.segmentation.post_buffer;
    max_gestures = params.segmentation.max_gestures;
    min_gap = params.segmentation.min_gap;
    
    % Convert durations to samples
    min_samples = round(min_duration * Fs);
    max_samples = round(max_duration * Fs);
    pre_samples = round(pre_buffer * Fs);
    post_samples = round(post_buffer * Fs);
    gap_samples = round(min_gap * Fs);
    
    %% Compute motion energy
    % Primary: gyroscope magnitude (rotation is key for gestures)
    gyr_mag = sqrt(sum(imu.gyr.^2, 2));
    
    % Secondary: accelerometer magnitude deviation from gravity
    acc_mag = sqrt(sum(imu.acc.^2, 2));
    acc_dev = abs(acc_mag - params.constants.g);
    
    % Combined energy (weighted)
    energy = gyr_mag + 0.3 * acc_dev;
    
    % Smooth energy signal
    window_size = max(3, round(0.05 * Fs));  % 50ms window
    energy_smooth = movmean(energy, window_size);
    
    seg.energy = energy_smooth;
    
    %% Hysteresis thresholding (state machine)
    state = zeros(n, 1);  % 0 = quiet, 1 = active
    
    % Start in quiet state
    current_state = 0;
    
    for k = 1:n
        if current_state == 0  % Quiet
            if energy_smooth(k) > energy_high
                current_state = 1;  % Transition to active
            end
        else  % Active
            if energy_smooth(k) < energy_low
                current_state = 0;  % Transition to quiet
            end
        end
        state(k) = current_state;
    end
    
    seg.state = state;
    
    %% Find contiguous active regions
    d = diff([0; state; 0]);
    starts = find(d == 1);
    ends = find(d == -1) - 1;
    
    n_raw = length(starts);
    fprintf('  Found %d raw active regions\n', n_raw);
    
    %% Filter by duration
    windows = [];
    
    for i = 1:n_raw
        duration = ends(i) - starts(i) + 1;
        
        if duration >= min_samples && duration <= max_samples
            % Add pre/post buffer
            win_start = max(1, starts(i) - pre_samples);
            win_end = min(n, ends(i) + post_samples);
            
            windows = [windows; win_start, win_end];
        end
    end
    
    %% Merge close windows
    if size(windows, 1) > 1
        merged = windows(1, :);
        
        for i = 2:size(windows, 1)
            if windows(i, 1) - merged(end, 2) < gap_samples
                % Merge with previous
                merged(end, 2) = windows(i, 2);
            else
                % Start new window
                merged = [merged; windows(i, :)];
            end
        end
        
        windows = merged;
    end
    
    %% Limit number of gestures
    if size(windows, 1) > max_gestures
        % Rank by energy and keep top ones
        energies = zeros(size(windows, 1), 1);
        for i = 1:size(windows, 1)
            energies(i) = sum(energy_smooth(windows(i,1):windows(i,2)));
        end
        
        [~, idx] = sort(energies, 'descend');
        windows = windows(idx(1:max_gestures), :);
        
        % Re-sort by time
        [~, idx] = sort(windows(:, 1));
        windows = windows(idx, :);
    end
    
    %% Store results
    seg.windows = windows;
    seg.n_gestures = size(windows, 1);
    
    fprintf('  Detected %d valid gestures\n', seg.n_gestures);
    
    %% Select primary gesture (highest energy)
    if seg.n_gestures > 0
        energies = zeros(seg.n_gestures, 1);
        for i = 1:seg.n_gestures
            energies(i) = sum(energy_smooth(windows(i,1):windows(i,2)));
        end
        
        [max_energy, primary] = max(energies);
        seg.primary = primary;
        seg.winIdx = windows(primary, :);
        
        % Compute confidence score based on energy prominence
        if seg.n_gestures > 1
            sorted_energies = sort(energies, 'descend');
            seg.score = max_energy / (max_energy + sorted_energies(2));
        else
            seg.score = 1.0;
        end
        
        fprintf('  Primary gesture: window %d (%.2f - %.2f s), score: %.2f\n', ...
            primary, imu.t(seg.winIdx(1)), imu.t(seg.winIdx(2)), seg.score);
    else
        seg.primary = 0;
        seg.winIdx = [1, n];  % Default to entire signal
        seg.score = 0;
        fprintf('  No gestures detected, using entire signal\n');
    end
    
    % Store time reference
    seg.t = imu.t;
    seg.Fs = Fs;
    
end
