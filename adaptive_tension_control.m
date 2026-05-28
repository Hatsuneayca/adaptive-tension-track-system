% =========================================================================
% PROJECT : AI-Supported Adaptive Tension Track System
% AUTHOR  : Ayça Bahar Güner — Dokuz Eylül University, Mech. Eng.
% VERSION : 5.0 — Full Edition (MC + Stability + PID Sweep + Time-Series)
% DATE    : May 2026
% =========================================================================

clear; clc; close all;

%% ══════════════════════════════════════════════════════════════════════════
%  1. CONFIGURATION
%% ══════════════════════════════════════════════════════════════════════════

% Time
time_step   = 0.01;
total_time  = 60;
time_vector = 0:time_step:total_time;
num_points  = length(time_vector);

% Physical constants
nominal_tension      = 50000;   % [N]
velocity_scaling     = 15;      % alpha
terrain_scaling      = 25000;   % beta
actuator_tau         = 0.3;     % [s] first-order plant time constant
tension_limit_upper  = 80000;   % [N]
tension_limit_lower  = 20000;   % [N]

% Nominal PID gains
kp = 1.2;
ki = 0.10;
kd = 0.08;
derivative_filter_pole = 50;    % Nf [rad/s]

% Actuator saturation
max_actuator_force =  15000;    % [N]
min_actuator_force = -15000;    % [N]

%% ══════════════════════════════════════════════════════════════════════════
%  2. SHARED HELPER: run one PID simulation, return tension trace + metrics
%% ══════════════════════════════════════════════════════════════════════════
% Defined as nested function at end of file — called below.

%% ══════════════════════════════════════════════════════════════════════════
%  3. MONTE CARLO ANALYSIS  (100 runs, nominal gains)
%% ══════════════════════════════════════════════════════════════════════════

num_runs              = 100;
rmse_results          = zeros(num_runs, 1);
derailment_risk       = zeros(num_runs, 1);

% Store one representative run for time-series plot
representative_actual = [];
representative_ref    = [];

fprintf('Running Monte Carlo (%d runs)...\n', num_runs);
rng('shuffle');

for run_index = 1:num_runs
    noise_intensity = 0.5 + 0.2*rand();
    [T_actual, T_ref] = run_pid_sim( ...
        time_vector, num_points, time_step, ...
        nominal_tension, velocity_scaling, terrain_scaling, ...
        actuator_tau, tension_limit_lower, tension_limit_upper, ...
        kp, ki, kd, ...
        max_actuator_force, min_actuator_force, ...
        noise_intensity);

    rmse_results(run_index)    = sqrt(mean((T_ref - T_actual).^2));
    derailment_risk(run_index) = 100 * sum(T_actual < tension_limit_lower) / num_points;

    if run_index == 1
        representative_actual = T_actual;
        representative_ref    = T_ref;
    end
end

%% ══════════════════════════════════════════════════════════════════════════
%  4. PID GAIN SWEEP  (vary Kp, observe RMSE — Ki/Kd held at nominal)
%% ══════════════════════════════════════════════════════════════════════════

kp_sweep       = linspace(0.3, 3.0, 20);
sweep_rmse     = zeros(size(kp_sweep));
rng(42);  % fixed seed so sweep is deterministic

fprintf('Running Kp sweep (%d values)...\n', length(kp_sweep));
for k = 1:length(kp_sweep)
    [T_actual, T_ref] = run_pid_sim( ...
        time_vector, num_points, time_step, ...
        nominal_tension, velocity_scaling, terrain_scaling, ...
        actuator_tau, tension_limit_lower, tension_limit_upper, ...
        kp_sweep(k), ki, kd, ...
        max_actuator_force, min_actuator_force, ...
        0.5);   % fixed noise for fair comparison
    sweep_rmse(k) = sqrt(mean((T_ref - T_actual).^2));
end

%% ══════════════════════════════════════════════════════════════════════════
%  5. FREQUENCY-DOMAIN STABILITY ANALYSIS
%% ══════════════════════════════════════════════════════════════════════════

s  = tf('s');
G  = 1 / (actuator_tau * s + 1);
C  = pid(kp, ki, kd, 1/derivative_filter_pole);
L  = C * G;
CL = feedback(L, 1);
[Gm, Pm, Wcg, Wcp] = margin(L);

%% ══════════════════════════════════════════════════════════════════════════
%  6. CONSOLE REPORT
%% ══════════════════════════════════════════════════════════════════════════

fprintf('\n═══════════════════════════════════════════\n');
fprintf('  ANALIZ OZETI — v5.0\n');
fprintf('═══════════════════════════════════════════\n');
fprintf('  Mean RMSE        : %8.1f N\n',   mean(rmse_results));
fprintf('  RMSE Std Dev     : %8.1f N\n',   std(rmse_results));
fprintf('  Derailment Risk  : %8.2f %%\n',  mean(derailment_risk));
fprintf('  Gain Margin      : %8.1f dB  @ %.2f rad/s\n', 20*log10(Gm), Wcg);
fprintf('  Phase Margin     : %8.1f deg @ %.2f rad/s\n', Pm, Wcp);
fprintf('  Optimal Kp (sweep): %.2f  → RMSE %.1f N\n', ...
        kp_sweep(sweep_rmse == min(sweep_rmse)), min(sweep_rmse));
fprintf('═══════════════════════════════════════════\n');

%% ══════════════════════════════════════════════════════════════════════════
%  7. VISUALISATION  (3×2 grid)
%% ══════════════════════════════════════════════════════════════════════════

figure('Name','Tension Track v5.0 — Full Analysis', ...
       'Position',[50 50 1400 900], 'Color','white');

%── 7.1  Time-series: actual vs reference (run 1)
subplot(3,2,1);
plot(time_vector, representative_ref/1000,    'b--', 'LineWidth', 1.2); hold on;
plot(time_vector, representative_actual/1000, 'r',   'LineWidth', 1.0);
yline(tension_limit_lower/1000, 'k:', 'LineWidth', 1.0);
yline(tension_limit_upper/1000, 'k:', 'LineWidth', 1.0);
legend('Reference','Actual','Limits','Location','best');
title('Time-Series: Tension Tracking (Run 1)');
xlabel('Time [s]'); ylabel('Tension [kN]'); grid on;

%── 7.2  Tracking error over time (run 1)
subplot(3,2,2);
tracking_error = (representative_ref - representative_actual) / 1000;
plot(time_vector, tracking_error, 'Color', [0.8 0.2 0.2], 'LineWidth', 1.0);
yline(0, 'k--');
title('Tracking Error (Run 1)');
xlabel('Time [s]'); ylabel('Error [kN]'); grid on;

%── 7.3  MC RMSE histogram
subplot(3,2,3);
histogram(rmse_results, 20, 'FaceColor',[0.2 0.6 0.8], 'EdgeColor','white');
hold on;
xline(mean(rmse_results), 'r--', 'LineWidth', 1.5, ...
      'Label', sprintf('Mean: %.0f N', mean(rmse_results)));
title('MC Robustness — RMSE Distribution');
xlabel('RMSE [N]'); ylabel('Count'); grid on;

%── 7.4  Kp sweep vs RMSE
subplot(3,2,4);
plot(kp_sweep, sweep_rmse, 'o-', 'Color',[0.2 0.7 0.3], 'LineWidth', 1.5, ...
     'MarkerFaceColor',[0.2 0.7 0.3]);
hold on;
[~, idx] = min(sweep_rmse);
plot(kp_sweep(idx), sweep_rmse(idx), 'r*', 'MarkerSize', 12);
xline(kp, 'b--', 'Label', 'Current Kp');
title('PID Gain Sweep: Kp vs RMSE');
xlabel('Kp'); ylabel('RMSE [N]'); grid on;

%── 7.5  Bode plot
subplot(3,2,5);
bode(L); 
title(sprintf('Open-Loop Bode | GM: %.1f dB  PM: %.1f°', 20*log10(Gm), Pm));
grid on;

%── 7.6  Closed-loop step response
subplot(3,2,6);
step(CL, 5);
title('Closed-Loop Step Response'); grid on;

%% ══════════════════════════════════════════════════════════════════════════
%  LOCAL FUNCTION — PID simulation (reused by MC loop and gain sweep)
%% ══════════════════════════════════════════════════════════════════════════

function [T_actual, T_ref] = run_pid_sim( ...
        time_vector, num_points, time_step, ...
        nominal_tension, velocity_scaling, terrain_scaling, ...
        actuator_tau, tension_limit_lower, tension_limit_upper, ...
        kp, ki, kd, ...
        max_force, min_force, noise_intensity)

    vehicle_velocity = 10 + 5*sin(0.1*time_vector) + noise_intensity*randn(1, num_points);
    terrain_friction = 0.5 + 0.3*sin(0.05*time_vector) + 0.05*randn(1, num_points);

    T_ref = nominal_tension + velocity_scaling*(vehicle_velocity.^2) - terrain_friction*terrain_scaling;
    T_ref = max(tension_limit_lower, min(tension_limit_upper, T_ref));

    T_actual     = zeros(1, num_points);
    T_actual(1)  = nominal_tension;
    error_integral = 0;
    prev_error     = 0;
    is_saturated   = false;

    for i = 2:num_points
        error            = T_ref(i) - T_actual(i-1);
        proportional     = kp * error;
        derivative       = kd * (error - prev_error) / time_step;

        if ~is_saturated
            error_integral = error_integral + error * time_step;
        end

        raw_signal     = proportional + (ki * error_integral) + derivative;
        applied_signal = max(min_force, min(max_force, raw_signal));
        is_saturated   = (raw_signal > max_force) || (raw_signal < min_force);

        dT           = (applied_signal - (T_actual(i-1) - nominal_tension)) / actuator_tau;
        T_actual(i)  = T_actual(i-1) + dT * time_step;
        prev_error   = error;
    end
end
