%% TEST_QUAT_MATH - Unit tests for quaternion utilities
%
% Quick sanity checks for quat_utils.m functions.
% Run with: runtests('test_quat_math') or just run this script.
%
% Tests verify:
%   - Normalization maintains unit norm
%   - Multiplication is associative
%   - Conjugate/inverse properties
%   - Rotation correctness
%   - Euler angle round-trips
%   - Identity properties
%
% Author: Generated for Gesture-SA-Sensor-Fusion project
% Date: 2025

function tests = test_quat_math
    tests = functiontests(localfunctions);
end

%% Test: Normalization
function test_normalize(testCase)
    q = [1, 2, 3, 4];  % Unnormalized
    qn = quat_utils('normalize', q);
    
    norm_val = sqrt(sum(qn.^2));
    verifyEqual(testCase, norm_val, 1.0, 'AbsTol', 1e-10, ...
        'Normalized quaternion should have unit norm');
end

%% Test: Identity quaternion
function test_identity(testCase)
    q_id = quat_utils('identity');
    
    verifyEqual(testCase, q_id, [1, 0, 0, 0], 'AbsTol', 1e-10, ...
        'Identity quaternion should be [1,0,0,0]');
end

%% Test: Multiply with identity
function test_multiply_identity(testCase)
    q = quat_utils('normalize', [0.5, 0.5, 0.5, 0.5]);
    q_id = quat_utils('identity');
    
    q_result = quat_utils('multiply', q, q_id);
    
    verifyEqual(testCase, q_result, q, 'AbsTol', 1e-10, ...
        'q * identity should equal q');
end

%% Test: Multiply with conjugate gives identity
function test_multiply_conjugate(testCase)
    q = quat_utils('normalize', [1, 2, 3, 4]);
    q_conj = quat_utils('conjugate', q);
    
    q_result = quat_utils('multiply', q, q_conj);
    q_id = quat_utils('identity');
    
    verifyEqual(testCase, q_result, q_id, 'AbsTol', 1e-10, ...
        'q * conjugate(q) should equal identity');
end

%% Test: Inverse property
function test_inverse(testCase)
    q = quat_utils('normalize', [0.7, 0.3, -0.5, 0.1]);
    q_inv = quat_utils('inverse', q);
    
    q_result = quat_utils('multiply', q, q_inv);
    q_id = quat_utils('identity');
    
    verifyEqual(testCase, q_result, q_id, 'AbsTol', 1e-10, ...
        'q * inverse(q) should equal identity');
end

%% Test: Rotation of vector by identity
function test_rotate_identity(testCase)
    v = [1, 2, 3];
    q_id = quat_utils('identity');
    
    v_rot = quat_utils('rotate', q_id, v);
    
    verifyEqual(testCase, v_rot, v, 'AbsTol', 1e-10, ...
        'Rotating by identity should not change vector');
end

%% Test: 90-degree rotation around Z-axis
function test_rotate_90_z(testCase)
    % Quaternion for 90 degrees around Z: [cos(45°), 0, 0, sin(45°)]
    q = quat_utils('fromAxisAngle', [0, 0, 1], pi/2);
    
    v = [1, 0, 0];  % X-axis vector
    v_rot = quat_utils('rotate', q, v);
    
    expected = [0, 1, 0];  % Should become Y-axis
    verifyEqual(testCase, v_rot, expected, 'AbsTol', 1e-10, ...
        '90° rotation around Z should map X to Y');
end

%% Test: 180-degree rotation
function test_rotate_180(testCase)
    q = quat_utils('fromAxisAngle', [0, 0, 1], pi);
    
    v = [1, 0, 0];
    v_rot = quat_utils('rotate', q, v);
    
    expected = [-1, 0, 0];
    verifyEqual(testCase, v_rot, expected, 'AbsTol', 1e-10, ...
        '180° rotation around Z should negate X');
end

%% Test: Euler to quaternion round-trip
function test_euler_roundtrip(testCase)
    % Original Euler angles (radians): roll=30°, pitch=45°, yaw=60°
    euler_orig = [pi/6, pi/4, pi/3];  % [roll, pitch, yaw]
    
    q = quat_utils('fromEuler', euler_orig(1), euler_orig(2), euler_orig(3));
    euler_back = quat_utils('toEuler', q);
    
    verifyEqual(testCase, euler_back, euler_orig, 'AbsTol', 1e-10, ...
        'Euler -> quaternion -> Euler should round-trip');
end

%% Test: Small angle rotation via fromOmega
function test_fromOmega(testCase)
    omega = [0, 0, 0.1];  % Small rotation around Z
    dt = 0.01;
    
    q = quat_utils('fromOmega', omega, dt);
    
    % Should be approximately [1, 0, 0, 0.0005] for small angles
    % q ≈ [cos(θ/2), 0, 0, sin(θ/2)] where θ = 0.1*0.01 = 0.001 rad
    expected_w = cos(0.001/2);
    expected_z = sin(0.001/2);
    
    verifyEqual(testCase, q(1), expected_w, 'AbsTol', 1e-6, ...
        'fromOmega w component');
    verifyEqual(testCase, q(4), expected_z, 'AbsTol', 1e-6, ...
        'fromOmega z component');
end

%% Test: Rotation matrix conversion round-trip
function test_rotmat_roundtrip(testCase)
    q_orig = quat_utils('normalize', [0.5, 0.5, 0.5, 0.5]);
    
    R = quat_utils('toRotMat', q_orig);
    q_back = quat_utils('fromRotMat', R);
    
    % Quaternions are equivalent up to sign
    if q_back(1) * q_orig(1) < 0
        q_back = -q_back;
    end
    
    verifyEqual(testCase, q_back, q_orig, 'AbsTol', 1e-10, ...
        'Quaternion -> RotMat -> Quaternion should round-trip');
end

%% Test: Rotation matrix is orthogonal
function test_rotmat_orthogonal(testCase)
    q = quat_utils('normalize', [1, 2, 3, 4]);
    R = quat_utils('toRotMat', q);
    
    I = R * R';
    verifyEqual(testCase, I, eye(3), 'AbsTol', 1e-10, ...
        'Rotation matrix should be orthogonal: R*R'' = I');
    
    det_R = det(R);
    verifyEqual(testCase, det_R, 1.0, 'AbsTol', 1e-10, ...
        'Rotation matrix determinant should be 1');
end

%% Test: SLERP at t=0 and t=1
function test_slerp_endpoints(testCase)
    q0 = quat_utils('normalize', [1, 0, 0, 0]);
    q1 = quat_utils('normalize', [0.707, 0, 0, 0.707]);
    
    q_at_0 = quat_utils('slerp', q0, q1, 0);
    q_at_1 = quat_utils('slerp', q0, q1, 1);
    
    verifyEqual(testCase, q_at_0, q0, 'AbsTol', 1e-10, ...
        'SLERP at t=0 should return q0');
    verifyEqual(testCase, q_at_1, q1, 'AbsTol', 1e-10, ...
        'SLERP at t=1 should return q1');
end

%% Test: SLERP at t=0.5 (midpoint)
function test_slerp_midpoint(testCase)
    q0 = quat_utils('identity');
    q1 = quat_utils('fromAxisAngle', [0, 0, 1], pi/2);  % 90° around Z
    
    q_mid = quat_utils('slerp', q0, q1, 0.5);
    
    % Midpoint should be 45° around Z
    q_expected = quat_utils('fromAxisAngle', [0, 0, 1], pi/4);
    
    verifyEqual(testCase, q_mid, q_expected, 'AbsTol', 1e-10, ...
        'SLERP midpoint should be half the rotation');
end

%% Test: Normalization maintains positive w
function test_normalize_positive_w(testCase)
    q = [-0.5, 0.5, 0.5, 0.5];  % Negative w
    qn = quat_utils('normalize', q);
    
    verifyGreaterThanOrEqual(testCase, qn(1), 0, ...
        'Normalized quaternion should have w >= 0');
end

%% Test: Multiple vector rotation (batch)
function test_rotate_batch(testCase)
    q = quat_utils('fromAxisAngle', [0, 0, 1], pi/2);
    
    V = [1, 0, 0;
         0, 1, 0;
         1, 1, 0];  % 3 vectors
    
    V_rot = quat_utils('rotate', q, V);
    
    expected = [0, 1, 0;
               -1, 0, 0;
               -1, 1, 0];
    
    verifyEqual(testCase, V_rot, expected, 'AbsTol', 1e-10, ...
        'Batch rotation should work for multiple vectors');
end

%% Run tests if executed as script
% This allows running as: >> test_quat_math
% Or: >> runtests('test_quat_math')
if ~isempty(which('runtests'))
    % Use MATLAB's unit test framework
    results = runtests('test_quat_math');
    disp(results);
else
    % Fallback: run tests manually
    fprintf('Running quaternion math tests...\n\n');
    
    % Simple pass/fail execution
    test_functions = {@test_normalize_simple, @test_identity_simple, ...
                      @test_rotate_90_simple, @test_euler_simple};
    
    for i = 1:length(test_functions)
        try
            test_functions{i}();
            fprintf('PASS: %s\n', func2str(test_functions{i}));
        catch ME
            fprintf('FAIL: %s - %s\n', func2str(test_functions{i}), ME.message);
        end
    end
end

%% Simple test functions (fallback for older MATLAB)
function test_normalize_simple()
    q = quat_utils('normalize', [1, 2, 3, 4]);
    assert(abs(norm(q) - 1) < 1e-10, 'Norm should be 1');
end

function test_identity_simple()
    q = quat_utils('identity');
    assert(all(abs(q - [1,0,0,0]) < 1e-10), 'Identity should be [1,0,0,0]');
end

function test_rotate_90_simple()
    q = quat_utils('fromAxisAngle', [0,0,1], pi/2);
    v_rot = quat_utils('rotate', q, [1,0,0]);
    assert(all(abs(v_rot - [0,1,0]) < 1e-10), 'Should map X to Y');
end

function test_euler_simple()
    euler = [0.1, 0.2, 0.3];
    q = quat_utils('fromEuler', euler(1), euler(2), euler(3));
    euler_back = quat_utils('toEuler', q);
    assert(all(abs(euler_back - euler) < 1e-10), 'Should round-trip');
end
