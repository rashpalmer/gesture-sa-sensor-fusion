function qu = quat_utils()
%QUAT_UTILS Collection of quaternion utility functions
%   qu = quat_utils() returns a struct of function handles for quaternion
%   operations.
%
%   QUATERNION CONVENTION:
%       q = [w, x, y, z] where w is the scalar part
%       This is the Hamilton convention
%       Unit quaternion ||q|| = 1 represents a rotation
%
%   AVAILABLE FUNCTIONS:
%       qu.normalize(q)           - Normalize quaternion to unit length
%       qu.multiply(q1, q2)       - Hamilton product q1 ⊗ q2
%       qu.conjugate(q)           - Quaternion conjugate q*
%       qu.inverse(q)             - Quaternion inverse (= conjugate for unit quat)
%       qu.rotate(q, v)           - Rotate vector v by quaternion q
%       qu.fromAxisAngle(axis,angle) - Create quaternion from axis-angle
%       qu.fromOmega(omega, dt)   - Create quaternion from angular velocity
%       qu.fromEuler(r, p, y)     - Create from Euler angles (ZYX convention)
%       qu.toEuler(q)             - Convert to Euler angles (ZYX convention)
%       qu.toRotMat(q)            - Convert to 3x3 rotation matrix
%       qu.fromRotMat(R)          - Create from rotation matrix
%       qu.slerp(q1, q2, t)       - Spherical linear interpolation
%       qu.identity()             - Identity quaternion [1,0,0,0]
%
%   EXAMPLE:
%       qu = quat_utils();
%       q = qu.fromAxisAngle([0;0;1], pi/2);  % 90° rotation about Z
%       v = [1; 0; 0];
%       v_rot = qu.rotate(q, v);  % Should be [0; 1; 0]
%
%   Author: Sensor Fusion Demo
%   Date: 2024

    % Return struct of function handles
    qu.normalize = @quatNormalize;
    qu.multiply = @quatMultiply;
    qu.conjugate = @quatConjugate;
    qu.inverse = @quatInverse;
    qu.rotate = @quatRotate;
    qu.fromAxisAngle = @quatFromAxisAngle;
    qu.fromOmega = @quatFromOmega;
    qu.fromEuler = @quatFromEuler;
    qu.toEuler = @quatToEuler;
    qu.toRotMat = @quatToRotMat;
    qu.fromRotMat = @quatFromRotMat;
    qu.slerp = @quatSlerp;
    qu.identity = @quatIdentity;
    qu.norm = @quatNorm;
    
end

%% ==================== FUNCTION IMPLEMENTATIONS ====================

function qn = quatNormalize(q)
%QUATNORMALIZE Normalize quaternion to unit length
%   q can be 4x1 or Nx4
    if size(q,2) == 4
        % Nx4 array
        norms = sqrt(sum(q.^2, 2));
        norms(norms < 1e-10) = 1;  % Prevent division by zero
        qn = q ./ norms;
    else
        % 4x1 vector
        n = norm(q);
        if n < 1e-10
            qn = [1; 0; 0; 0];
        else
            qn = q / n;
        end
    end
end

function n = quatNorm(q)
%QUATNORM Compute quaternion norm
    if size(q,2) == 4
        n = sqrt(sum(q.^2, 2));
    else
        n = norm(q);
    end
end

function q = quatIdentity()
%QUATIDENTITY Return identity quaternion
    q = [1; 0; 0; 0];
end

function qc = quatConjugate(q)
%QUATCONJUGATE Compute quaternion conjugate
%   For q = [w, x, y, z], q* = [w, -x, -y, -z]
    if size(q,2) == 4
        qc = [q(:,1), -q(:,2), -q(:,3), -q(:,4)];
    else
        qc = [q(1); -q(2); -q(3); -q(4)];
    end
end

function qi = quatInverse(q)
%QUATINVERSE Compute quaternion inverse
%   For unit quaternions, inverse = conjugate
%   General: q^-1 = q* / ||q||²
    if size(q,2) == 4
        qc = quatConjugate(q);
        n2 = sum(q.^2, 2);
        n2(n2 < 1e-10) = 1;
        qi = qc ./ n2;
    else
        qc = quatConjugate(q);
        n2 = sum(q.^2);
        if n2 < 1e-10
            qi = quatIdentity();
        else
            qi = qc / n2;
        end
    end
end

function q12 = quatMultiply(q1, q2)
%QUATMULTIPLY Hamilton quaternion product q1 ⊗ q2
%   q1, q2: 4x1 quaternions [w; x; y; z]
%   Result represents rotation q1 followed by q2 (in that order)
%
%   Hamilton product formula:
%   (w1 + x1*i + y1*j + z1*k) * (w2 + x2*i + y2*j + z2*k)
    
    if size(q1,2) == 4 && size(q2,2) == 4
        % Both are Nx4
        w1 = q1(:,1); x1 = q1(:,2); y1 = q1(:,3); z1 = q1(:,4);
        w2 = q2(:,1); x2 = q2(:,2); y2 = q2(:,3); z2 = q2(:,4);
        
        q12 = [w1.*w2 - x1.*x2 - y1.*y2 - z1.*z2, ...
               w1.*x2 + x1.*w2 + y1.*z2 - z1.*y2, ...
               w1.*y2 - x1.*z2 + y1.*w2 + z1.*x2, ...
               w1.*z2 + x1.*y2 - y1.*x2 + z1.*w2];
    else
        % Handle 4x1 vectors
        if size(q1,2) == 4, q1 = q1'; end
        if size(q2,2) == 4, q2 = q2'; end
        
        w1 = q1(1); x1 = q1(2); y1 = q1(3); z1 = q1(4);
        w2 = q2(1); x2 = q2(2); y2 = q2(3); z2 = q2(4);
        
        q12 = [w1*w2 - x1*x2 - y1*y2 - z1*z2;
               w1*x2 + x1*w2 + y1*z2 - z1*y2;
               w1*y2 - x1*z2 + y1*w2 + z1*x2;
               w1*z2 + x1*y2 - y1*x2 + z1*w2];
    end
end

function v_rot = quatRotate(q, v)
%QUATROTATE Rotate vector v by quaternion q
%   q: 4x1 unit quaternion
%   v: 3x1 or 3xN vector(s)
%   
%   Formula: v' = q ⊗ [0; v] ⊗ q*
%   Optimized implementation avoiding full quaternion multiply
    
    if size(q,2) == 4, q = q'; end  % Ensure column
    
    % Extract quaternion components
    w = q(1);
    qv = q(2:4);  % Vector part
    
    if size(v,1) ~= 3
        v = v';  % Ensure 3xN
    end
    
    % Rodrigues rotation formula (optimized)
    % v' = v + 2*w*(qv × v) + 2*(qv × (qv × v))
    for i = 1:size(v,2)
        t = 2 * cross(qv, v(:,i));
        v_rot(:,i) = v(:,i) + w*t + cross(qv, t);
    end
end

function q = quatFromAxisAngle(axis, angle)
%QUATFROMAXISANGLE Create quaternion from axis-angle representation
%   axis: 3x1 unit vector (rotation axis)
%   angle: scalar (rotation angle in radians)
%
%   q = [cos(θ/2), sin(θ/2)*axis]
    
    if size(axis,2) > 1, axis = axis'; end
    
    % Normalize axis
    axis_norm = norm(axis);
    if axis_norm < 1e-10
        q = quatIdentity();
        return;
    end
    axis = axis / axis_norm;
    
    half_angle = angle / 2;
    q = [cos(half_angle); sin(half_angle) * axis];
end

function q = quatFromOmega(omega, dt)
%QUATFROMOMEGA Create quaternion from angular velocity
%   omega: 3x1 angular velocity (rad/s)
%   dt: time step (seconds)
%
%   This represents the rotation that occurs over time dt
%   at constant angular velocity omega.
%
%   For small angles: q ≈ [1, omega*dt/2]
%   Exact: q = exp(omega*dt/2) in quaternion space
    
    if size(omega,2) > 1, omega = omega'; end
    
    angle = norm(omega) * dt;
    
    if angle < 1e-10
        % Small angle approximation
        q = quatNormalize([1; omega * dt / 2]);
    else
        % Full computation
        axis = omega / norm(omega);
        q = quatFromAxisAngle(axis, angle);
    end
end

function q = quatFromEuler(roll, pitch, yaw)
%QUATFROMEULER Create quaternion from Euler angles (ZYX convention)
%   roll: rotation about X axis (radians)
%   pitch: rotation about Y axis (radians)
%   yaw: rotation about Z axis (radians)
%
%   ZYX means: first yaw, then pitch, then roll
%   This is aerospace convention (phi, theta, psi)
    
    cr = cos(roll/2);  sr = sin(roll/2);
    cp = cos(pitch/2); sp = sin(pitch/2);
    cy = cos(yaw/2);   sy = sin(yaw/2);
    
    q = [cr*cp*cy + sr*sp*sy;
         sr*cp*cy - cr*sp*sy;
         cr*sp*cy + sr*cp*sy;
         cr*cp*sy - sr*sp*cy];
    
    q = quatNormalize(q);
end

function [roll, pitch, yaw] = quatToEuler(q)
%QUATTOEULER Convert quaternion to Euler angles (ZYX convention)
%   q: 4x1 quaternion or Nx4 array
%   Returns roll, pitch, yaw in radians
%
%   WARNING: Has singularity at pitch = ±90°
    
    if size(q,1) == 4 && size(q,2) == 1
        q = q';  % Convert to 1x4
    end
    
    if size(q,2) ~= 4
        error('Quaternion must be 4x1 or Nx4');
    end
    
    w = q(:,1); x = q(:,2); y = q(:,3); z = q(:,4);
    
    % Roll (X-axis rotation)
    sinr_cosp = 2 * (w.*x + y.*z);
    cosr_cosp = 1 - 2 * (x.^2 + y.^2);
    roll = atan2(sinr_cosp, cosr_cosp);
    
    % Pitch (Y-axis rotation)
    sinp = 2 * (w.*y - z.*x);
    % Clamp to avoid numerical issues at poles
    sinp = max(min(sinp, 1), -1);
    pitch = asin(sinp);
    
    % Yaw (Z-axis rotation)
    siny_cosp = 2 * (w.*z + x.*y);
    cosy_cosp = 1 - 2 * (y.^2 + z.^2);
    yaw = atan2(siny_cosp, cosy_cosp);
    
    % Return as column vectors if input was single quaternion
    if length(roll) == 1
        roll = roll(1);
        pitch = pitch(1);
        yaw = yaw(1);
    end
end

function R = quatToRotMat(q)
%QUATTOROTMAT Convert quaternion to 3x3 rotation matrix
%   q: 4x1 unit quaternion [w; x; y; z]
%   R: 3x3 rotation matrix
%
%   Vector rotation: v' = R * v
    
    if size(q,2) == 4, q = q'; end
    q = quatNormalize(q);
    
    w = q(1); x = q(2); y = q(3); z = q(4);
    
    R = [1-2*(y^2+z^2),   2*(x*y-w*z),   2*(x*z+w*y);
         2*(x*y+w*z),   1-2*(x^2+z^2),   2*(y*z-w*x);
         2*(x*z-w*y),     2*(y*z+w*x), 1-2*(x^2+y^2)];
end

function q = quatFromRotMat(R)
%QUATFROMROTMAT Create quaternion from rotation matrix
%   R: 3x3 rotation matrix
%   q: 4x1 unit quaternion
%
%   Uses Shepperd's method for numerical stability
    
    % Trace
    tr = R(1,1) + R(2,2) + R(3,3);
    
    if tr > 0
        S = sqrt(tr + 1) * 2;
        w = 0.25 * S;
        x = (R(3,2) - R(2,3)) / S;
        y = (R(1,3) - R(3,1)) / S;
        z = (R(2,1) - R(1,2)) / S;
    elseif R(1,1) > R(2,2) && R(1,1) > R(3,3)
        S = sqrt(1 + R(1,1) - R(2,2) - R(3,3)) * 2;
        w = (R(3,2) - R(2,3)) / S;
        x = 0.25 * S;
        y = (R(1,2) + R(2,1)) / S;
        z = (R(1,3) + R(3,1)) / S;
    elseif R(2,2) > R(3,3)
        S = sqrt(1 + R(2,2) - R(1,1) - R(3,3)) * 2;
        w = (R(1,3) - R(3,1)) / S;
        x = (R(1,2) + R(2,1)) / S;
        y = 0.25 * S;
        z = (R(2,3) + R(3,2)) / S;
    else
        S = sqrt(1 + R(3,3) - R(1,1) - R(2,2)) * 2;
        w = (R(2,1) - R(1,2)) / S;
        x = (R(1,3) + R(3,1)) / S;
        y = (R(2,3) + R(3,2)) / S;
        z = 0.25 * S;
    end
    
    q = quatNormalize([w; x; y; z]);
    
    % Ensure w >= 0 for consistency
    if q(1) < 0
        q = -q;
    end
end

function q = quatSlerp(q1, q2, t)
%QUATSLERP Spherical linear interpolation between quaternions
%   q1, q2: 4x1 unit quaternions
%   t: interpolation parameter [0, 1]
%      t=0 returns q1, t=1 returns q2
    
    if size(q1,2) == 4, q1 = q1'; end
    if size(q2,2) == 4, q2 = q2'; end
    
    q1 = quatNormalize(q1);
    q2 = quatNormalize(q2);
    
    % Compute dot product
    dot_prod = sum(q1 .* q2);
    
    % If negative dot product, negate one quaternion
    % (quaternions q and -q represent the same rotation)
    if dot_prod < 0
        q2 = -q2;
        dot_prod = -dot_prod;
    end
    
    % Clamp for numerical stability
    dot_prod = min(max(dot_prod, -1), 1);
    
    % Compute interpolation
    theta = acos(dot_prod);
    
    if abs(theta) < 1e-6
        % Linear interpolation for small angles
        q = quatNormalize((1-t)*q1 + t*q2);
    else
        % SLERP
        q = (sin((1-t)*theta)/sin(theta))*q1 + (sin(t*theta)/sin(theta))*q2;
        q = quatNormalize(q);
    end
end
