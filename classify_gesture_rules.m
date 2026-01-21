function cls = classify_gesture_rules(feat, params)
%CLASSIFY_GESTURE_RULES Rule-based gesture classifier using feature thresholds
%
%   cls = CLASSIFY_GESTURE_RULES(feat, params)
%
%   Rule-based classification using decision tree logic based on extracted
%   features. This serves as a transparent baseline before ML approaches.
%
%   INPUTS:
%       feat   - Feature struct from extract_features.m containing:
%                .x      : 1xM numeric feature vector
%                .names  : 1xM cell array of feature names
%                .values : Struct with named feature values
%
%       params - Configuration from config_params.m (optional)
%
%   OUTPUTS:
%       cls    - Classification result struct:
%                .label    : String gesture label (e.g., "twist", "shake")
%                .score    : Confidence score [0, 1]
%                .method   : "rules" (identifies classifier type)
%                .reason   : Human-readable explanation
%                .matches  : Struct with rule match details per gesture
%                .features : Copy of key features used in decision
%
%   GESTURE SIGNATURES (expected sensor patterns):
%   -----------------------------------------------
%   1. TWIST (yaw rotation about vertical axis)
%      - High gyro_z (yaw rate), low gyro_x/y
%      - Moderate duration (0.3-1.5s)
%      - Low linear acceleration deviation
%
%   2. FLIP_UP (pitch rotation, screen faces up at end)
%      - High positive gyro_x (pitch up)
%      - Moderate duration
%      - Gravity vector shifts from -Z toward -Y
%
%   3. FLIP_DOWN (pitch rotation, screen faces down at end)
%      - High negative gyro_x (pitch down)
%      - Similar to flip_up but opposite sign
%
%   4. SHAKE (lateral oscillation)
%      - High RMS gyro (especially y-axis if phone held vertically)
%      - Multiple zero-crossings
%      - High frequency content
%      - Short-medium duration
%
%   5. PUSH_FORWARD (linear thrust along phone's long axis)
%      - High peak linear acceleration in body Y
%      - Low rotation overall
%      - Very short duration
%
%   6. CIRCLE (circular motion in a plane)
%      - Sustained rotation in multiple axes
%      - Gyro_x and gyro_y correlated with phase shift
%      - Longer duration
%      - Cross-correlation lag indicates circular pattern
%
%   USAGE EXAMPLE:
%       params = config_params();
%       data = read_phone_data('gesture_log.mat');
%       imu = preprocess_imu(data, params);
%       seg = segment_gesture(imu, params);
%       feat = extract_features(imu, seg, [], params);
%       cls = classify_gesture_rules(feat, params);
%       fprintf('Detected: %s (%.0f%% confidence)\n', cls.label, cls.score*100);
%       fprintf('Reason: %s\n', cls.reason);
%
%   See also: EXTRACT_FEATURES, SEGMENT_GESTURE, ML_PREDICT_BASELINE
%
%   Reference: This implements a hierarchical decision tree approach
%   where dominant axis and feature magnitudes determine classification.
%
%   Author: Claude (Anthropic) - Sensor Fusion Demonstrator
%   Date: January 2026

    %% Input validation
    if nargin < 1 || isempty(feat)
        error('classify_gesture_rules:NoInput', 'Feature struct required.');
    end
    
    if nargin < 2 || isempty(params)
        params = config_params();
    end
    
    % Validate feature struct
    if ~isfield(feat, 'values') || ~isstruct(feat.values)
        error('classify_gesture_rules:InvalidFeatures', ...
              'feat.values struct required.');
    end
    
    %% Extract key features from struct
    v = feat.values;  % Shorthand
    
    % Get feature values with defaults for missing fields
    duration      = getFieldOrDefault(v, 'duration', 0.5);
    
    % Gyroscope features
    gyr_rms_x     = getFieldOrDefault(v, 'gyr_rms_x', 0);
    gyr_rms_y     = getFieldOrDefault(v, 'gyr_rms_y', 0);
    gyr_rms_z     = getFieldOrDefault(v, 'gyr_rms_z', 0);
    gyr_peak_x    = getFieldOrDefault(v, 'gyr_peak_x', 0);
    gyr_peak_y    = getFieldOrDefault(v, 'gyr_peak_y', 0);
    gyr_peak_z    = getFieldOrDefault(v, 'gyr_peak_z', 0);
    gyr_peak_abs  = getFieldOrDefault(v, 'gyr_peak_abs', 0);
    dominant_axis = getFieldOrDefault(v, 'dominant_axis', 1);
    total_rot     = getFieldOrDefault(v, 'total_rotation_deg', 0);
    
    % Zero-crossings
    zc_gyr_x      = getFieldOrDefault(v, 'zc_gyr_x', 0);
    zc_gyr_y      = getFieldOrDefault(v, 'zc_gyr_y', 0);
    zc_gyr_z      = getFieldOrDefault(v, 'zc_gyr_z', 0);
    zc_total      = zc_gyr_x + zc_gyr_y + zc_gyr_z;
    
    % Accelerometer features
    acc_rms_x     = getFieldOrDefault(v, 'acc_rms_x', 0);
    acc_rms_y     = getFieldOrDefault(v, 'acc_rms_y', 0);
    acc_rms_z     = getFieldOrDefault(v, 'acc_rms_z', 0);
    acc_range_x   = getFieldOrDefault(v, 'acc_range_x', 0);
    acc_range_y   = getFieldOrDefault(v, 'acc_range_y', 0);
    acc_range_z   = getFieldOrDefault(v, 'acc_range_z', 0);
    acc_rms_total = sqrt(acc_rms_x^2 + acc_rms_y^2 + acc_rms_z^2);
    
    % Orientation change features (if available)
    delta_roll    = getFieldOrDefault(v, 'delta_roll', 0);
    delta_pitch   = getFieldOrDefault(v, 'delta_pitch', 0);
    delta_yaw     = getFieldOrDefault(v, 'delta_yaw', 0);
    
    % Phase features (cross-correlation for circular detection)
    xcorr_lag     = getFieldOrDefault(v, 'xcorr_lag_xy', 0);
    
    % Energy ratios
    gyr_energy_ratio_xy = getFieldOrDefault(v, 'gyr_energy_ratio_xy', 1);
    gyr_energy_ratio_z  = getFieldOrDefault(v, 'gyr_energy_ratio_z', 0);
    
    %% Get thresholds from params
    if isfield(params, 'gestures') && isfield(params.gestures, 'rules')
        rules = params.gestures.rules;
    else
        % Default thresholds
        rules = struct();
        rules.twist_min_gyr_z_rms = 1.5;      % rad/s
        rules.twist_max_xy_ratio = 0.5;        % z should dominate
        rules.flip_min_gyr_rms = 2.0;          % rad/s
        rules.flip_min_delta_pitch = 45;       % degrees
        rules.shake_min_zc = 4;                % zero-crossings
        rules.shake_min_gyr_rms = 1.5;         % rad/s
        rules.push_min_acc_range = 5.0;        % m/s²
        rules.push_max_gyr_rms = 1.0;          % rad/s (low rotation)
        rules.circle_min_duration = 0.8;       % seconds
        rules.circle_min_lag = 5;              % samples phase shift
        rules.circle_min_rotation = 180;       % degrees
    end
    
    %% Initialize match scores for each gesture
    % Each gesture gets a score based on how well features match expected pattern
    matches = struct();
    gestures = {'twist', 'flip_up', 'flip_down', 'shake', 'push_forward', 'circle', 'unknown'};
    
    for i = 1:length(gestures)
        matches.(gestures{i}) = struct('score', 0, 'reasons', {{}});
    end
    
    %% Rule evaluation for each gesture type
    
    % =====================================================================
    % TWIST: Yaw rotation (gyro_z dominant)
    % =====================================================================
    twist_score = 0;
    twist_reasons = {};
    
    if gyr_rms_z > rules.twist_min_gyr_z_rms
        twist_score = twist_score + 0.3;
        twist_reasons{end+1} = sprintf('High gyro_z RMS (%.2f rad/s)', gyr_rms_z);
    end
    
    gyr_xy_rms = sqrt(gyr_rms_x^2 + gyr_rms_y^2);
    if gyr_rms_z > 0 && (gyr_xy_rms / gyr_rms_z) < rules.twist_max_xy_ratio
        twist_score = twist_score + 0.3;
        twist_reasons{end+1} = 'Z-axis dominates rotation';
    end
    
    if dominant_axis == 3  % Z is dominant
        twist_score = twist_score + 0.2;
        twist_reasons{end+1} = 'Dominant axis is Z (yaw)';
    end
    
    if abs(delta_yaw) > 30 && abs(delta_yaw) > abs(delta_pitch) && abs(delta_yaw) > abs(delta_roll)
        twist_score = twist_score + 0.2;
        twist_reasons{end+1} = sprintf('Large yaw change (%.1f°)', delta_yaw);
    end
    
    matches.twist.score = min(1.0, twist_score);
    matches.twist.reasons = twist_reasons;
    
    % =====================================================================
    % FLIP_UP: Pitch rotation with positive direction
    % =====================================================================
    flip_up_score = 0;
    flip_up_reasons = {};
    
    if gyr_rms_x > rules.flip_min_gyr_rms && gyr_peak_x > 0
        flip_up_score = flip_up_score + 0.4;
        flip_up_reasons{end+1} = sprintf('Strong positive pitch (peak %.2f rad/s)', gyr_peak_x);
    end
    
    if dominant_axis == 1 && gyr_peak_x > 0
        flip_up_score = flip_up_score + 0.2;
        flip_up_reasons{end+1} = 'X-axis dominant with positive peak';
    end
    
    if delta_pitch > rules.flip_min_delta_pitch
        flip_up_score = flip_up_score + 0.3;
        flip_up_reasons{end+1} = sprintf('Pitch increased by %.1f°', delta_pitch);
    end
    
    if zc_gyr_x <= 2  % Single flip should have few zero-crossings
        flip_up_score = flip_up_score + 0.1;
        flip_up_reasons{end+1} = 'Low zero-crossings (single motion)';
    end
    
    matches.flip_up.score = min(1.0, flip_up_score);
    matches.flip_up.reasons = flip_up_reasons;
    
    % =====================================================================
    % FLIP_DOWN: Pitch rotation with negative direction
    % =====================================================================
    flip_down_score = 0;
    flip_down_reasons = {};
    
    if gyr_rms_x > rules.flip_min_gyr_rms && gyr_peak_x < 0
        flip_down_score = flip_down_score + 0.4;
        flip_down_reasons{end+1} = sprintf('Strong negative pitch (peak %.2f rad/s)', gyr_peak_x);
    end
    
    if dominant_axis == 1 && gyr_peak_x < 0
        flip_down_score = flip_down_score + 0.2;
        flip_down_reasons{end+1} = 'X-axis dominant with negative peak';
    end
    
    if delta_pitch < -rules.flip_min_delta_pitch
        flip_down_score = flip_down_score + 0.3;
        flip_down_reasons{end+1} = sprintf('Pitch decreased by %.1f°', abs(delta_pitch));
    end
    
    if zc_gyr_x <= 2
        flip_down_score = flip_down_score + 0.1;
        flip_down_reasons{end+1} = 'Low zero-crossings (single motion)';
    end
    
    matches.flip_down.score = min(1.0, flip_down_score);
    matches.flip_down.reasons = flip_down_reasons;
    
    % =====================================================================
    % SHAKE: Oscillatory motion with multiple zero-crossings
    % =====================================================================
    shake_score = 0;
    shake_reasons = {};
    
    if zc_total >= rules.shake_min_zc
        shake_score = shake_score + 0.4;
        shake_reasons{end+1} = sprintf('High zero-crossings (%d total)', zc_total);
    end
    
    gyr_rms_total = sqrt(gyr_rms_x^2 + gyr_rms_y^2 + gyr_rms_z^2);
    if gyr_rms_total > rules.shake_min_gyr_rms
        shake_score = shake_score + 0.3;
        shake_reasons{end+1} = sprintf('High gyro RMS (%.2f rad/s)', gyr_rms_total);
    end
    
    if duration < 1.0 && duration > 0.2
        shake_score = shake_score + 0.1;
        shake_reasons{end+1} = sprintf('Short duration (%.2fs)', duration);
    end
    
    % Check for oscillation pattern (similar energy in both directions)
    if gyr_energy_ratio_xy > 0.3 && gyr_energy_ratio_xy < 3.0
        shake_score = shake_score + 0.2;
        shake_reasons{end+1} = 'Balanced X-Y gyro energy (oscillation)';
    end
    
    matches.shake.score = min(1.0, shake_score);
    matches.shake.reasons = shake_reasons;
    
    % =====================================================================
    % PUSH_FORWARD: Linear acceleration with minimal rotation
    % =====================================================================
    push_score = 0;
    push_reasons = {};
    
    % High acceleration range (push involves linear motion)
    max_acc_range = max([acc_range_x, acc_range_y, acc_range_z]);
    if max_acc_range > rules.push_min_acc_range
        push_score = push_score + 0.4;
        push_reasons{end+1} = sprintf('High acc range (%.2f m/s²)', max_acc_range);
    end
    
    % Low rotation (push should be mostly linear)
    if gyr_rms_total < rules.push_max_gyr_rms
        push_score = push_score + 0.3;
        push_reasons{end+1} = sprintf('Low rotation (%.2f rad/s RMS)', gyr_rms_total);
    end
    
    % Typically along phone's Y-axis (long axis)
    if acc_range_y > acc_range_x && acc_range_y > acc_range_z
        push_score = push_score + 0.2;
        push_reasons{end+1} = 'Y-axis dominant acceleration (along phone)';
    end
    
    % Short duration
    if duration < 0.6
        push_score = push_score + 0.1;
        push_reasons{end+1} = sprintf('Quick motion (%.2fs)', duration);
    end
    
    matches.push_forward.score = min(1.0, push_score);
    matches.push_forward.reasons = push_reasons;
    
    % =====================================================================
    % CIRCLE: Sustained circular motion
    % =====================================================================
    circle_score = 0;
    circle_reasons = {};
    
    % Longer duration for drawing circle
    if duration > rules.circle_min_duration
        circle_score = circle_score + 0.2;
        circle_reasons{end+1} = sprintf('Sustained motion (%.2fs)', duration);
    end
    
    % Large total rotation
    if total_rot > rules.circle_min_rotation
        circle_score = circle_score + 0.3;
        circle_reasons{end+1} = sprintf('Large total rotation (%.1f°)', total_rot);
    end
    
    % Cross-correlation lag indicates phase shift between X and Y gyro
    % (characteristic of circular motion)
    if abs(xcorr_lag) > rules.circle_min_lag
        circle_score = circle_score + 0.3;
        circle_reasons{end+1} = sprintf('X-Y gyro phase shift detected (lag=%d)', xcorr_lag);
    end
    
    % Energy in multiple axes
    if gyr_rms_x > 0.5 && gyr_rms_y > 0.5
        circle_score = circle_score + 0.2;
        circle_reasons{end+1} = 'Multi-axis rotation sustained';
    end
    
    matches.circle.score = min(1.0, circle_score);
    matches.circle.reasons = circle_reasons;
    
    %% Find best match
    % Get all scores
    scores = [matches.twist.score, matches.flip_up.score, matches.flip_down.score, ...
              matches.shake.score, matches.push_forward.score, matches.circle.score];
    labels = {'twist', 'flip_up', 'flip_down', 'shake', 'push_forward', 'circle'};
    
    [max_score, max_idx] = max(scores);
    
    %% Determine final classification
    min_confidence = 0.3;  % Minimum score to make a classification
    
    if max_score < min_confidence
        % No gesture matched well enough
        cls.label = 'unknown';
        cls.score = 1 - max_score;  % Confidence in "unknown"
        cls.reason = sprintf('No gesture matched (best was %s at %.0f%%)', ...
                            labels{max_idx}, max_score * 100);
    else
        cls.label = labels{max_idx};
        cls.score = max_score;
        
        % Build reason string from matched reasons
        matched_reasons = matches.(cls.label).reasons;
        if isempty(matched_reasons)
            cls.reason = sprintf('Classified as %s', cls.label);
        else
            cls.reason = strjoin(matched_reasons, '; ');
        end
    end
    
    %% Check for ambiguous cases (two gestures score similarly)
    scores_sorted = sort(scores, 'descend');
    if length(scores_sorted) >= 2 && (scores_sorted(1) - scores_sorted(2)) < 0.15
        % Find second-best match
        second_idx = find(scores == scores_sorted(2), 1);
        cls.reason = [cls.reason, sprintf(' (Note: %s also scored %.0f%%)', ...
                     labels{second_idx}, scores_sorted(2) * 100)];
    end
    
    %% Populate output struct
    cls.method = 'rules';
    cls.matches = matches;
    cls.features = struct(...
        'duration', duration, ...
        'gyr_rms_total', gyr_rms_total, ...
        'gyr_peak_abs', gyr_peak_abs, ...
        'dominant_axis', dominant_axis, ...
        'total_rotation', total_rot, ...
        'zc_total', zc_total, ...
        'acc_rms_total', acc_rms_total ...
    );
    
    %% Print summary if verbose
    if isfield(params, 'verbose') && params.verbose
        fprintf('\n=== Rule-Based Classification ===\n');
        fprintf('Result: %s (%.0f%% confidence)\n', cls.label, cls.score * 100);
        fprintf('Reason: %s\n', cls.reason);
        fprintf('Scores: twist=%.2f, flip_up=%.2f, flip_down=%.2f, ', ...
                matches.twist.score, matches.flip_up.score, matches.flip_down.score);
        fprintf('shake=%.2f, push=%.2f, circle=%.2f\n', ...
                matches.shake.score, matches.push_forward.score, matches.circle.score);
    end
end

%% Helper function to safely get struct field
function val = getFieldOrDefault(s, fieldname, default)
%GETFIELDORDEFAULT Get field value or return default if missing
    if isfield(s, fieldname)
        val = s.(fieldname);
        if isempty(val) || (isnumeric(val) && isnan(val))
            val = default;
        end
    else
        val = default;
    end
end
