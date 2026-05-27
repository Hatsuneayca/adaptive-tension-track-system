% =========================================================================
% PROJECT : AI-Supported Adaptive Tension Track System
% AUTHOR  : Ayça Bahar Güner — Dokuz Eylül University, Mech. Eng.
% VERSION : 2.0 — Polished PoC with dimensional consistency & realistic terrain
% DATE    : May 2026
%
% NOTES ON PLANT MODEL:
%   The actuator is modelled as a first-order lag (tau = 0.3 s), consistent
%   with typical hydraulic cylinder response times reported in literature
%   (e.g., Ryu et al., 2010, J. Terramechanics). This is a deliberate
%   simplification; the full nonlinear sprocket-engagement model is
%   deferred to the ANSYS / Simulink Digital Twin phase.
%
% DIMENSIONAL ANALYSIS OF REFERENCE TENSION EQUATION:
%   T_ref = T0 + alpha*(v^2) - beta*mu       [Units: N]
%   T0    : N        (nominal pre-tension)
%   v     : m/s      → v^2 : m^2/s^2
%   alpha : N·s^2/m^2  (empirical; to be replaced by GA/PSO optimisation)
%   mu    : dimensionless
%   beta  : N        (empirical; same note)
% =========================================================================

clear; clc; close all;

%% ── 1. SIMULATION PARAMETERS ────────────────────────────────────────────
dt    = 0.01;           % [s]  timestep — reduced for smoother dynamics
t_end = 60;             % [s]  total run time
t     = 0:dt:t_end;
N     = length(t);

%% ── 2. PHYSICAL & CONTROLLER CONSTANTS ──────────────────────────────────
% --- Reference tension model (dimensionally consistent) ---
T0    = 50000;          % [N]       nominal pre-tension
alpha = 15;             % [N·s²/m²] velocity weight   (heuristic — see note above)
beta  = 25000;          % [N]       terrain weight     (heuristic — see note above)

% --- PID gains (tuned for tau=0.3 plant, dt=0.01) ---
Kp = 1.2;
Ki = 0.10;
Kd = 0.08;

% --- Actuator physical limits ---
u_max =  15000;         % [N]  max hydraulic force increment
u_min = -15000;         % [N]  min hydraulic force increment

% --- First-order plant time constant ---
tau   = 0.3;            % [s]  hydraulic cylinder lag (literature-based)

%% ── 3. SYNTHETIC SENSOR DATA ─────────────────────────────────────────────
% Vehicle speed: sinusoidal manoeuvre + low Gaussian noise
rng(42);                % fixed seed → reproducible results
v_vehicle = 10 + 5*sin(0.1*t) + 0.5*randn(1,N);   % [m/s]

% Terrain friction: smooth transitions via sigmoid (physically realistic)
% Three zones: asphalt (μ≈0.8) → mud (μ≈0.3) → rocky (μ≈0.6)
mu_asphalt = 0.8;
mu_mud     = 0.3;
mu_rocky   = 0.6;
k_sig      = 0.8;       % sigmoid steepness — controls transition sharpness [1/s]

mu_terrain = mu_asphalt ./ (1 + exp( k_sig*(t - 20))) + ...
             mu_mud     ./ (1 + exp(-k_sig*(t - 20))) ./ (1 + exp( k_sig*(t - 40))) * ...
             (1 + exp(-k_sig*(t - 20))) + ...   % blend correction
             mu_rocky   ./ (1 + exp(-k_sig*(t - 40)));

% Simpler, more readable sigmoid blend:
w1 = 1 ./ (1 + exp( k_sig*(t - 20)));           % weight for asphalt
w2 = 1 ./ (1 + exp(-k_sig*(t - 20))) .* ...
     1 ./ (1 + exp( k_sig*(t - 40)));            % weight for mud
w3 = 1 ./ (1 + exp(-k_sig*(t - 40)));            % weight for rocky

mu_terrain = w1*mu_asphalt + w2*mu_mud + w3*mu_rocky;
mu_terrain = mu_terrain + 0.015*randn(1,N);      % sensor noise [dimensionless]

%% ── 4. REFERENCE TENSION (Physics-Based Model / Future: ML Output) ───────
T_ref = T0 + alpha*(v_vehicle.^2) - beta*mu_terrain;   % [N]

% Clamp reference to physically meaningful range
T_min_physical = 20000;   % [N]  below this → derailment risk
T_max_physical = 80000;   % [N]  above this → excessive bushing wear
T_ref = max(T_min_physical, min(T_max_physical, T_ref));

%% ── 5. ANTI-WINDUP PID CONTROL LOOP ──────────────────────────────────────
T_actual  = zeros(1,N);
T_actual(1) = T0;

u_history = zeros(1,N);
e         = zeros(1,N);
e_int     = 0;
windup_flag = zeros(1,N);   % diagnostic: track when clamping fires

for i = 2:N
    % Error
    e(i) = T_ref(i) - T_actual(i-1);

    % PD terms
    P = Kp * e(i);
    D = Kd * (e(i) - e(i-1)) / dt;

    % Anti-windup: conditional integration (clamping method)
    u_test = P + Ki*e_int + D;
    saturated_high = (u_test >= u_max) && (e(i) > 0);
    saturated_low  = (u_test <= u_min) && (e(i) < 0);

    if saturated_high || saturated_low
        windup_flag(i) = 1;   % integrator frozen
    else
        e_int = e_int + e(i)*dt;
    end

    I = Ki * e_int;
    u = P + I + D;
    u = max(u_min, min(u_max, u));   % saturate actuator output

    u_history(i) = u;

    % Plant: first-order lag
    % dT/dt = (u - (T_actual - T0)) / tau
    % Euler forward integration
    dT = (u - (T_actual(i-1) - T0)) / tau;
    T_actual(i) = T_actual(i-1) + dT*dt;
end

%% ── 6. PERFORMANCE METRICS ───────────────────────────────────────────────
RMSE        = sqrt(mean(e.^2));
max_error   = max(abs(e));
windup_pct  = 100 * sum(windup_flag) / N;

% Derailment risk: timesteps where T_actual < T_min_physical
derail_risk_pct = 100 * sum(T_actual < T_min_physical) / N;

fprintf('═══════════════════════════════════════\n');
fprintf('  SIMULATION PERFORMANCE METRICS\n');
fprintf('═══════════════════════════════════════\n');
fprintf('  RMSE (tracking error)   : %8.1f N\n', RMSE);
fprintf('  Max absolute error      : %8.1f N\n', max_error);
fprintf('  Anti-windup active      : %8.1f %%\n', windup_pct);
fprintf('  Derailment risk window  : %8.1f %%\n', derail_risk_pct);
fprintf('═══════════════════════════════════════\n');

%% ── 7. VISUALISATION ─────────────────────────────────────────────────────
fig = figure('Name','Adaptive Track Tension — PoC v2.0', ...
             'Position',[80, 40, 1000, 900], ...
             'Color','white');

% Subplot 1: Vehicle speed
ax1 = subplot(5,1,1);
plot(t, v_vehicle, 'Color',[0.1 0.4 0.8], 'LineWidth',1.4);
ylabel('Speed (m/s)');
title('Vehicle Kinematics — IMU Sensor Input');
grid on; xlim([0 t_end]);

% Subplot 2: Terrain friction (with zone annotations)
ax2 = subplot(5,1,2);
plot(t, mu_terrain, 'Color',[0.8 0.2 0.1], 'LineWidth',1.4);
ylabel('\mu [ – ]');
title('Terrain Friction Coefficient — Smooth Sigmoid Transitions');
yline(mu_asphalt,'b--','Asphalt','LabelHorizontalAlignment','left','FontSize',8);
yline(mu_mud,    'g--','Mud',    'LabelHorizontalAlignment','left','FontSize',8);
yline(mu_rocky,  'k--','Rocky',  'LabelHorizontalAlignment','left','FontSize',8);
grid on; xlim([0 t_end]); ylim([0.1 1.0]);

% Subplot 3: Tension tracking
ax3 = subplot(5,1,3);
plot(t, T_ref,    'k--', 'LineWidth',1.8, 'DisplayName','T_{ref} (target)'); hold on;
plot(t, T_actual, 'Color',[0.1 0.7 0.2], 'LineWidth',1.5, 'DisplayName','T_{actual}');
yline(T_min_physical,'r:','Derailment threshold','FontSize',8,'LabelHorizontalAlignment','left');
ylabel('Tension (N)');
title(sprintf('PID Tracking — RMSE = %.0f N', RMSE));
legend('Location','best','FontSize',8); grid on; xlim([0 t_end]);

% Subplot 4: Control signal
ax4 = subplot(5,1,4);
plot(t, u_history, 'Color',[0.6 0.1 0.7], 'LineWidth',1.4);
yline( u_max,'r--','u_{max}','LabelHorizontalAlignment','left','FontSize',8);
yline( u_min,'r--','u_{min}','LabelHorizontalAlignment','left','FontSize',8);
ylabel('Force (N)');
title('Actuator Control Signal — Hydraulic Valve Command');
grid on; xlim([0 t_end]); ylim([u_min*1.2 u_max*1.2]);

% Subplot 5: Anti-windup diagnostic
ax5 = subplot(5,1,5);
area(t, windup_flag*u_max, 'FaceColor',[1 0.6 0.1], 'FaceAlpha',0.5, 'EdgeColor','none');
ylabel('Active [0/1]');
xlabel('Time (s)');
title(sprintf('Anti-Windup Clamping Events — Active %.1f%% of runtime', windup_pct));
grid on; xlim([0 t_end]); ylim([-1000 u_max*1.1]);

linkaxes([ax1 ax2 ax3 ax4 ax5], 'x');   % synchronized x-axis zoom

sgtitle('AI-Supported Adaptive Track Tension System — PoC v2.0', ...
        'FontSize',13,'FontWeight','bold');
