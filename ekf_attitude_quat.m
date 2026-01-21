function est = ekf_attitude_quat(imu, params)
%EKF_ATTITUDE_QUAT Extended Kalman Filter for quaternion attitude estimation
%   est = ekf_attitude_quat(imu, params) estimates device orientation using
%   an EKF that fuses gyroscope, accelerometer, and magnetometer data.
%
%   INPUTS:
%       imu     - Preprocessed IMU data from preprocess_imu()
%       params  - Configuration from config_params() (optional)
%
%   OUTPUT:
%       est - Estimation results struct:
%           .q          - Nx4 quaternion trajectory [w,x,y,z]
%           .b_g        - Nx3 estimated gyro bias (rad/s)
%           .euler      - Nx3 Euler angles [roll,pitch,yaw] (rad) for plotting
%           .P          - 7x7xN covariance history
%           .Ptrace     - Nx7 diagonal covariance traces
%           .innov_acc  - Nx3 accelerometer innovations
%           .innov_mag  - Nx3 magnetometer innovations
%           .S_acc      - Nx3 innovation covariances (acc)
%           .params     - Parameters used
%
%   STATE VECTOR (7 states):
%       x = [q_w, q_x, q_y, q_z, b_gx, b_gy, b_gz]'
%
%   PROCESS MODEL:
%       Quaternion: q_{k+1} = q_k ⊗ Δq(ω_m - b_g, dt)
%       Gyro bias:  b_{k+1} = b_k + w_b (random walk)
%
%   MEASUREMENT MODELS:
%       Gravity:    z_g = R(q)' * g_world    (from accelerometer)
%       Magnetic:   z_m = R(q)' * m_world    (from magnetometer)
%
%   EXAMPLE:
%       imu = preprocess_imu(data, params);
%       est = ekf_attitude_quat(imu, params);
%       plot(imu.t, est.euler * 180/pi);
%       legend('Roll', 'Pitch', 'Yaw');
%
%   REFERENCES:
%       - Quaternion kinematics for the error-state Kalman filter (Sola, 2017)
%       - MathWorks: ahrsfilter, imufilter documentation
%
%   Author: Sensor Fusion Demo
%   Date: 2024

    %% Default parameters
    if nargin < 2
        params = config_params();
    end
    
    fprintf('Running EKF attitude estimation...\n');
    
    %% Initialize
    qu = quat_utils();  % Quaternion utilities
    
    n = length(imu.t);
    dt_vec = [imu.dt; imu.dt(end)];  % Extend to n samples
    
    % State dimension
    n_states = 7;  % 4 quaternion + 3 gyro bias
    
    % Allocate output arrays
    est.q = zeros(n, 4);
    est.b_g = zeros(n, 3);
    est.P = zeros(n_states, n_states, n);
    est.Ptrace = zeros(n, n_states);
    est.innov_acc = zeros(n, 3);
    est.innov_mag = zeros(n, 3);
    est.S_acc = zeros(n, 3);
    
    %% Initialize state and covariance
    % Initial quaternion (identity - aligned with world frame)
    % Could estimate from first few samples using TRIAD or similar
    q0 = initialize_orientation(imu, params);
    
    % Initial gyro bias
    b_g0 = params.ekf.init_gyro_bias;
    if isfield(imu, 'calib') && isfield(imu.calib, 'gyro_bias')
        b_g0 = imu.calib.gyro_bias;
    end
    
    % Initial state
    x = [q0; b_g0];
    
    % Initial covariance
    P = diag([params.ekf.P0_quat * ones(4,1); 
              params.ekf.P0_bias * ones(3,1)]);
    
    %% Process and measurement noise covariances
    % Process noise Q (7x7)
    Q = diag([params.ekf.Q_quat * ones(4,1);
              params.ekf.Q_gyro_bias * ones(3,1)]);
    
    % Measurement noise R
    R_acc = params.ekf.R_acc * eye(3);
    R_mag = params.ekf.R_mag * eye(3);
    
    %% Reference vectors in world frame
    g_world = params.frames.gravity_world;  % Gravity direction
    m_world = params.frames.mag_ref_world;  % Magnetic north direction
    m_world = m_world / norm(m_world);      % Normalize
    
    %% Main EKF Loop
    for k = 1:n
        %% Store current estimates
        est.q(k, :) = x(1:4)';
        est.b_g(k, :) = x(5:7)';
        est.P(:,:,k) = P;
        est.Ptrace(k, :) = diag(P)';
        
        if k == n
            break;  % No prediction after last sample
        end
        
        %% Get measurements and dt
        dt = dt_vec(k);
        omega_m = imu.gyr(k, :)';  % Measured angular velocity
        acc_m = imu.acc(k, :)';    % Measured acceleration
        mag_m = imu.mag(k, :)';    % Measured magnetic field
        
        %% === PREDICTION STEP ===
        
        % Extract current state
        q = x(1:4);
        b_g = x(5:7);
        
        % Corrected angular velocity
        omega = omega_m - b_g;
        
        % Quaternion derivative: q_dot = 0.5 * q ⊗ [0; omega]
        % Propagate quaternion using exponential map
        q_delta = qu.fromOmega(omega, dt);
        q_pred = qu.multiply(q, q_delta);
        q_pred = qu.normalize(q_pred);
        
        % Gyro bias prediction (random walk - no change)
        b_g_pred = b_g;
        
        % Predicted state
        x_pred = [q_pred; b_g_pred];
        
        % State transition Jacobian F = df/dx
        F = compute_state_jacobian(q, omega, dt);
        
        % Predicted covariance
        P_pred = F * P * F' + Q;
        
        %% === MEASUREMENT UPDATE (Accelerometer) ===
        
        % Only update if acceleration magnitude is reasonable
        % (close to gravity, not during high-G maneuvers)
        acc_mag = norm(acc_m);
        acc_window = params.ekf.acc_magnitude_window;
        
        if acc_mag > acc_window(1) && acc_mag < acc_window(2)
            % Predicted gravity in body frame
            R_pred = qu.toRotMat(q_pred);
            g_pred = R_pred' * g_world;
            
            % Normalize acceleration to gravity direction
            acc_norm = acc_m / acc_mag * params.constants.g;
            
            % Innovation (measurement residual)
            y_acc = acc_norm - g_pred;
            
            % Measurement Jacobian H_acc = dh/dx
            H_acc = compute_acc_jacobian(q_pred, g_world);
            
            % Innovation covariance
            S_acc = H_acc * P_pred * H_acc' + R_acc;
            
            % Kalman gain
            K_acc = P_pred * H_acc' / S_acc;
            
            % State update
            x_upd = x_pred + K_acc * y_acc;
            
            % Covariance update (Joseph form for stability)
            I_KH = eye(n_states) - K_acc * H_acc;
            P_upd = I_KH * P_pred * I_KH' + K_acc * R_acc * K_acc';
            
            % Store innovations
            est.innov_acc(k, :) = y_acc';
            est.S_acc(k, :) = diag(S_acc)';
            
        else
            % Skip accelerometer update
            x_upd = x_pred;
            P_upd = P_pred;
        end
        
        %% === MEASUREMENT UPDATE (Magnetometer) ===
        
        if params.ekf.use_mag_update && ~all(isnan(mag_m)) && ...
           ~imu.flags.mag_outlier(k)
            
            q_current = x_upd(1:4);
            
            % Predicted magnetic field in body frame
            R_current = qu.toRotMat(q_current);
            
            % Project world magnetic reference to current heading
            % (we only use horizontal components for yaw correction)
            m_pred = R_current' * m_world;
            
            % Normalize magnetometer measurement
            mag_norm = mag_m / norm(mag_m);
            
            % Innovation
            y_mag = mag_norm - m_pred;
            
            % Check for outliers
            if norm(y_mag) < params.ekf.mag_rejection_threshold * sqrt(params.ekf.R_mag)
                
                % Measurement Jacobian H_mag
                H_mag = compute_mag_jacobian(q_current, m_world);
                
                % Innovation covariance
                S_mag = H_mag * P_upd * H_mag' + R_mag;
                
                % Kalman gain
                K_mag = P_upd * H_mag' / S_mag;
                
                % State update
                x_upd = x_upd + K_mag * y_mag;
                
                % Covariance update
                I_KH = eye(n_states) - K_mag * H_mag;
                P_upd = I_KH * P_upd * I_KH' + K_mag * R_mag * K_mag';
                
                est.innov_mag(k, :) = y_mag';
            end
        end
        
        %% === NORMALIZATION ===
        
        % Re-normalize quaternion
        x_upd(1:4) = qu.normalize(x_upd(1:4));
        
        % Ensure positive scalar part (quaternion sign convention)
        if x_upd(1) < 0
            x_upd(1:4) = -x_upd(1:4);
        end
        
        %% Prepare for next iteration
        x = x_upd;
        P = P_upd;
        
    end
    
    %% Post-processing
    
    % Convert to Euler angles for visualization
    est.euler = zeros(n, 3);
    for k = 1:n
        [roll, pitch, yaw] = qu.toEuler(est.q(k,:)');
        est.euler(k, :) = [roll, pitch, yaw];
    end
    
    % Store parameters used
    est.params = params.ekf;
    est.t = imu.t;
    
    %% Summary
    fprintf('EKF complete:\n');
    fprintf('  Final quaternion: [%.3f, %.3f, %.3f, %.3f]\n', est.q(end,:));
    fprintf('  Final gyro bias:  [%.4f, %.4f, %.4f] rad/s\n', est.b_g(end,:));
    fprintf('  Quaternion norm range: [%.6f, %.6f]\n', ...
        min(sqrt(sum(est.q.^2, 2))), max(sqrt(sum(est.q.^2, 2))));
    
end

%% ==================== HELPER FUNCTIONS ====================

function q0 = initialize_orientation(imu, params)
%INITIALIZE_ORIENTATION Estimate initial orientation from static data
    
    qu = quat_utils();
    
    % Find first static segment
    static_idx = find(imu.flags.stationary, 50);
    
    if length(static_idx) >= 10
        % Use TRIAD algorithm with accel and mag
        acc_init = mean(imu.acc(static_idx, :))';
        mag_init = mean(imu.mag(static_idx, :))';
        
        if ~all(isnan(acc_init)) && ~all(isnan(mag_init))
            q0 = triad_quaternion(acc_init, mag_init, params);
            return;
        end
    end
    
    % Fallback: use first sample
    acc_init = imu.acc(1, :)';
    
    if ~all(isnan(acc_init))
        % Estimate tilt from gravity
        acc_init = acc_init / norm(acc_init);
        
        roll = atan2(-acc_init(2), -acc_init(3));
        pitch = atan2(acc_init(1), sqrt(acc_init(2)^2 + acc_init(3)^2));
        yaw = 0;  % Cannot determine without magnetometer
        
        q0 = qu.fromEuler(roll, pitch, yaw);
    else
        % Identity quaternion
        q0 = [1; 0; 0; 0];
    end
end

function q = triad_quaternion(acc, mag, params)
%TRIAD_QUATERNION Compute orientation from accelerometer and magnetometer
    
    qu = quat_utils();
    
    % Body frame vectors
    v1b = -acc / norm(acc);  % Gravity direction (down)
    v2b = mag / norm(mag);   % Magnetic field
    
    % World frame vectors
    v1w = -params.frames.gravity_world / norm(params.frames.gravity_world);
    v2w = params.frames.mag_ref_world / norm(params.frames.mag_ref_world);
    
    % Construct orthonormal triads
    w1b = v1b;
    w2b = cross(v1b, v2b); w2b = w2b / norm(w2b);
    w3b = cross(w1b, w2b);
    
    w1w = v1w;
    w2w = cross(v1w, v2w); w2w = w2w / norm(w2w);
    w3w = cross(w1w, w2w);
    
    % Rotation matrices
    Mb = [w1b, w2b, w3b];
    Mw = [w1w, w2w, w3w];
    
    % Body to world rotation
    R = Mw * Mb';
    
    % Convert to quaternion
    q = qu.fromRotMat(R);
end

function F = compute_state_jacobian(q, omega, dt)
%COMPUTE_STATE_JACOBIAN Compute state transition Jacobian
    
    % State: [q_w, q_x, q_y, q_z, b_gx, b_gy, b_gz]
    
    % Quaternion derivative matrix
    % dq/dt = 0.5 * Omega(omega) * q
    % where Omega is the skew-symmetric matrix form
    
    wx = omega(1); wy = omega(2); wz = omega(3);
    
    % Omega matrix (for Hamilton convention)
    Omega = [0, -wx, -wy, -wz;
             wx,  0,  wz, -wy;
             wy, -wz,  0,  wx;
             wz,  wy, -wx,  0];
    
    % Discrete approximation: F_q ≈ I + 0.5*dt*Omega
    F_q = eye(4) + 0.5 * dt * Omega;
    
    % Jacobian of quaternion w.r.t. gyro bias
    % d(q_new)/d(b_g) = -0.5 * dt * Gamma(q)
    qw = q(1); qx = q(2); qy = q(3); qz = q(4);
    
    Gamma = [-qx, -qy, -qz;
              qw, -qz,  qy;
              qz,  qw, -qx;
             -qy,  qx,  qw];
    
    F_qb = -0.5 * dt * Gamma;
    
    % Gyro bias transition (identity - random walk)
    F_b = eye(3);
    
    % Full Jacobian
    F = [F_q,       F_qb;
         zeros(3,4), F_b];
end

function H = compute_acc_jacobian(q, g_world)
%COMPUTE_ACC_JACOBIAN Measurement Jacobian for accelerometer
    
    % Measurement: h(x) = R(q)' * g_world
    % Need: dh/dq
    
    qu = quat_utils();
    
    qw = q(1); qx = q(2); qy = q(3); qz = q(4);
    gx = g_world(1); gy = g_world(2); gz = g_world(3);
    
    % Partial derivatives of R(q)'*g with respect to quaternion components
    % Using chain rule and rotation matrix formula
    
    H_q = 2 * [qw*gx + qz*gy - qy*gz,  qx*gx + qy*gy + qz*gz, -qy*gx + qx*gy - qw*gz, -qz*gx + qw*gy + qx*gz;
               -qz*gx + qw*gy + qx*gz,  qy*gx - qx*gy + qw*gz,  qx*gx + qy*gy + qz*gz, -qw*gx - qz*gy + qy*gz;
               qy*gx - qx*gy + qw*gz,  qz*gx - qw*gy - qx*gz,  qw*gx + qz*gy - qy*gz,  qx*gx + qy*gy + qz*gz];
    
    % No dependence on gyro bias
    H_b = zeros(3, 3);
    
    H = [H_q, H_b];
end

function H = compute_mag_jacobian(q, m_world)
%COMPUTE_MAG_JACOBIAN Measurement Jacobian for magnetometer
    
    % Same structure as accelerometer
    qw = q(1); qx = q(2); qy = q(3); qz = q(4);
    mx = m_world(1); my = m_world(2); mz = m_world(3);
    
    H_q = 2 * [qw*mx + qz*my - qy*mz,  qx*mx + qy*my + qz*mz, -qy*mx + qx*my - qw*mz, -qz*mx + qw*my + qx*mz;
               -qz*mx + qw*my + qx*mz,  qy*mx - qx*my + qw*mz,  qx*mx + qy*my + qz*mz, -qw*mx - qz*my + qy*mz;
               qy*mx - qx*my + qw*mz,  qz*mx - qw*my - qx*mz,  qw*mx + qz*my - qy*mz,  qx*mx + qy*my + qz*mz];
    
    H_b = zeros(3, 3);
    
    H = [H_q, H_b];
end
