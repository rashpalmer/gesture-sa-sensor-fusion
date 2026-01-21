function tu = time_utils()
%TIME_UTILS Collection of time-related utility functions
%   tu = time_utils() returns a struct of function handles
%
%   AVAILABLE FUNCTIONS:
%       tu.computeDt(t)           - Compute time differences
%       tu.resampleUniform(t, x, Fs) - Resample to uniform rate
%       tu.findStaticSegments(data, params) - Find stationary periods
%       tu.getTimestamp()         - Get current timestamp string
%
%   Author: Sensor Fusion Demo

    tu.computeDt = @computeDt;
    tu.resampleUniform = @resampleUniform;
    tu.findStaticSegments = @findStaticSegments;
    tu.getTimestamp = @getTimestamp;
    
end

function [dt, Fs_mean, Fs_var] = computeDt(t)
%COMPUTEDT Compute time differences and sample rate statistics
%   t: Nx1 time vector (seconds)
%   dt: (N-1)x1 time differences
%   Fs_mean: mean sample rate
%   Fs_var: variance in sample rate
    
    t = t(:);  % Ensure column
    dt = diff(t);
    
    % Remove outliers for statistics
    dt_clean = dt(dt > 0 & dt < 0.5);  % Assume Fs > 2 Hz
    
    if isempty(dt_clean)
        Fs_mean = NaN;
        Fs_var = NaN;
    else
        Fs_mean = 1 / mean(dt_clean);
        Fs_var = var(1 ./ dt_clean);
    end
end

function [t_new, x_new] = resampleUniform(t, x, Fs_target)
%RESAMPLEUNIFORM Resample signal to uniform sample rate
%   t: Nx1 time vector (may be non-uniform)
%   x: NxM data matrix (N samples, M channels)
%   Fs_target: desired sample rate (Hz)
%
%   Returns uniformly sampled data using linear interpolation

    t = t(:);
    
    % Create uniform time vector
    t_new = (t(1) : 1/Fs_target : t(end))';
    
    % Interpolate each channel
    if size(x, 1) ~= length(t)
        x = x';  % Transpose if needed
    end
    
    n_channels = size(x, 2);
    x_new = zeros(length(t_new), n_channels);
    
    for ch = 1:n_channels
        x_new(:, ch) = interp1(t, x(:, ch), t_new, 'linear', 'extrap');
    end
end

function [static_idx, static_windows] = findStaticSegments(gyr, params)
%FINDSTATICSEGMENTS Identify stationary periods from gyroscope data
%   gyr: Nx3 gyroscope data (rad/s)
%   params: configuration parameters
%
%   static_idx: Nx1 logical array (true = static)
%   static_windows: Mx2 matrix of [start_idx, end_idx] for each segment

    if nargin < 2
        threshold = 0.5;  % rad/s
        min_samples = 50;
    else
        threshold = params.preprocess.static_threshold;
        min_samples = params.preprocess.static_window;
    end
    
    % Compute gyro magnitude
    gyr_mag = sqrt(sum(gyr.^2, 2));
    
    % Initial classification
    static_idx = gyr_mag < threshold;
    
    % Apply morphological operations to clean up
    % Dilate then erode (closing operation)
    se = ones(min_samples, 1);
    static_idx = movmean(double(static_idx), min_samples) > 0.5;
    
    % Find contiguous segments
    d = diff([0; static_idx(:); 0]);
    starts = find(d == 1);
    ends = find(d == -1) - 1;
    
    % Filter by minimum duration
    valid = (ends - starts + 1) >= min_samples;
    static_windows = [starts(valid), ends(valid)];
    
    % Convert back to logical array
    static_idx = false(size(gyr_mag));
    for i = 1:size(static_windows, 1)
        static_idx(static_windows(i,1):static_windows(i,2)) = true;
    end
end

function ts = getTimestamp()
%GETTIMESTAMP Get current timestamp as string for logging
    ts = datestr(now, 'yyyymmdd_HHMMSS');
end
