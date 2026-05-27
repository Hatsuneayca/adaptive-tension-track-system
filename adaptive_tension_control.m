% =========================================================================
% PROJECT : AI-Supported Adaptive Tension Track System
% AUTHOR  : Ayça Bahar Güner — Dokuz Eylül University, Mech. Eng.
% VERSION : 3.0 — Full PoC + Frequency Domain Stability Analysis
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
%   T0    : N          (nominal pre-tension)
%   v     : m/s   → v^2 : m^2/s^2
%   alpha : N·s^2/m^2  (empirical; to be replaced by GA/PSO optimisation)
%   mu    : dimensionless
%   beta  : N          (empirical; same note)
% =========================================================================

clear; clc; close all;

%% ── 1. SIMULATION PARAMETERS ────────────────────────────────────────────
dt    = 0.01;           % [s]  timestep
t_end = 60;             % [s]  total run time
t     = 0:dt:t_end;
N     = length(t);

%% ── 2. PHYSICAL & CONTROLLER CONSTANTS ──────────────────────────────────
T0    = 50000;          % [N]       nominal pre-tension
alpha = 15;             % [N·s²/m²] velocity weight   (heuristic)
beta  = 25000;          % [N]       terrain weight     (heuristic)

Kp = 1.2;
Ki = 0.10;
Kd = 0.08;

u_max =  15000;         % [N]  max hydraulic force increment
u_min = -15000;         % [N]  min hydraulic force increment

tau   = 0.3;            % [s]  hydraulic cylinder lag (literature-based)

%% ── 3. SYNTHETIC SENSOR DATA ─────────────────────────────────────────────
rng(42);
v_vehicle = 10 + 5*sin(0.1*t) + 0.5*randn(1,N);   % [m/s]

mu_asphalt = 0.8;
mu_mud     = 0.3;
mu_rocky   = 0.6;
k_sig      = 0.8;

w1 = 1 ./ (1 + exp( k_sig*(t - 20)));
w2 = 1 ./ (1 + exp(-k_sig*(t - 20))) .* ...
     1 ./ (1 + exp( k_sig*(t - 40)));
w3 = 1 ./ (1 + exp(-k_sig*(t - 40)));

mu_terrain = w1*mu_asphalt + w2*mu_mud + w3*mu_rocky;
mu_terrain = mu_terrain + 0.015*randn(1,N);

%% ── 4. REFERENCE TENSION ─────────────────────────────────────────────────
T_ref = T0 + alpha*(v_vehicle.^2) - beta*mu_terrain;

T_min_physical = 20000;
T_max_physical = 80000;
T_ref = max(T_min_physical, min(T_max_physical, T_ref));

%% ── 5. ANTI-WINDUP PID CONTROL LOOP ──────────────────────────────────────
T_actual    = zeros(1,N);
T_actual(1) = T0;
u_history   = zeros(1,N);
e           = zeros(1,N);
e_int       = 0;
windup_flag = zeros(1,N);

for i = 2:N
    e(i) = T_ref(i) - T_actual(i-1);

    P = Kp * e(i);
    D = Kd * (e(i) - e(i-1)) / dt;

    u_test         = P + Ki*e_int + D;
    saturated_high = (u_test >= u_max) && (e(i) > 0);
    saturated_low  = (u_test <= u_min) && (e(i) < 0);

    if saturated_high || saturated_low
        windup_flag(i) = 1;
    else
        e_int = e_int + e(i)*dt;
    end

    I = Ki * e_int;
    u = max(u_min, min(u_max, P + I + D));
    u_history(i) = u;

    dT = (u - (T_actual(i-1) - T0)) / tau;
    T_actual(i) = T_actual(i-1) + dT*dt;
end

%% ── 6. PERFORMANCE METRICS ───────────────────────────────────────────────
RMSE            = sqrt(mean(e.^2));
max_error       = max(abs(e));
windup_pct      = 100 * sum(windup_flag) / N;
derail_risk_pct = 100 * sum(T_actual < T_min_physical) / N;

fprintf('═══════════════════════════════════════\n');
fprintf('  SIMULATION PERFORMANCE METRICS\n');
fprintf('═══════════════════════════════════════\n');
fprintf('  RMSE (tracking error)   : %8.1f N\n', RMSE);
fprintf('  Max absolute error      : %8.1f N\n', max_error);
fprintf('  Anti-windup active      : %8.1f %%\n', windup_pct);
fprintf('  Derailment risk window  : %8.1f %%\n', derail_risk_pct);
fprintf('═══════════════════════════════════════\n');

%% ── 7. TIME DOMAIN VISUALISATION ─────────────────────────────────────────
figure('Name','Adaptive Track Tension — Time Domain', ...
       'Position',[50, 40, 1000, 900], 'Color','white');

ax1 = subplot(5,1,1);
plot(t, v_vehicle, 'Color',[0.1 0.4 0.8], 'LineWidth',1.4);
ylabel('Speed (m/s)'); title('Vehicle Kinematics — IMU Sensor Input');
grid on; xlim([0 t_end]);

ax2 = subplot(5,1,2);
plot(t, mu_terrain, 'Color',[0.8 0.2 0.1], 'LineWidth',1.4);
ylabel('\mu [ – ]'); title('Terrain Friction — Sigmoid Transitions');
yline(mu_asphalt,'b--','Asphalt','LabelHorizontalAlignment','left','FontSize',8);
yline(mu_mud,    'g--','Mud',    'LabelHorizontalAlignment','left','FontSize',8);
yline(mu_rocky,  'k--','Rocky',  'LabelHorizontalAlignment','left','FontSize',8);
grid on; xlim([0 t_end]); ylim([0.1 1.0]);

ax3 = subplot(5,1,3);
plot(t, T_ref,    'k--', 'LineWidth',1.8, 'DisplayName','T_{ref}'); hold on;
plot(t, T_actual, 'Color',[0.1 0.7 0.2], 'LineWidth',1.5, 'DisplayName','T_{actual}');
yline(T_min_physical,'r:','Derailment threshold','FontSize',8,'LabelHorizontalAlignment','left');
ylabel('Tension (N)'); title(sprintf('PID Tracking — RMSE = %.0f N', RMSE));
legend('Location','best','FontSize',8); grid on; xlim([0 t_end]);

ax4 = subplot(5,1,4);
plot(t, u_history, 'Color',[0.6 0.1 0.7], 'LineWidth',1.4);
yline(u_max,'r--','u_{max}','LabelHorizontalAlignment','left','FontSize',8);
yline(u_min,'r--','u_{min}','LabelHorizontalAlignment','left','FontSize',8);
ylabel('Force (N)'); title('Actuator Control Signal');
grid on; xlim([0 t_end]); ylim([u_min*1.2 u_max*1.2]);

ax5 = subplot(5,1,5);
area(t, windup_flag*u_max, 'FaceColor',[1 0.6 0.1], 'FaceAlpha',0.5, 'EdgeColor','none');
ylabel('Active [0/1]'); xlabel('Time (s)');
title(sprintf('Anti-Windup Events — Active %.1f%% of runtime', windup_pct));
grid on; xlim([0 t_end]); ylim([-1000 u_max*1.1]);

linkaxes([ax1 ax2 ax3 ax4 ax5], 'x');
sgtitle('AI-Supported Adaptive Track Tension — Time Domain (v3.0)', ...
        'FontSize',13,'FontWeight','bold');

%% ── 8. FREQUENCY DOMAIN — STABILITY ANALYSIS ────────────────────────────
s        = tf('s');
G        = 1 / (tau*s + 1);
N_filter = 100;
C        = pid(Kp, Ki, Kd, 1/N_filter);
L        = C * G;
CL       = feedback(L, 1);

[Gm, Pm, Wcg, Wcp] = margin(L);
bw = bandwidth(CL);

fprintf('\n═══════════════════════════════════════\n');
fprintf('  STABILITY MARGINS\n');
fprintf('═══════════════════════════════════════\n');
fprintf('  Gain margin          : %6.2f dB   (at %.2f rad/s)\n', 20*log10(Gm), Wcg);
fprintf('  Phase margin         : %6.2f deg  (at %.2f rad/s)\n', Pm, Wcp);
fprintf('  Closed-loop BW       : %6.2f rad/s  (%.2f Hz)\n', bw, bw/(2*pi));
fprintf('  Stability assessment : ');
if Pm > 45 && 20*log10(Gm) > 6
    fprintf('ROBUST (Pm>45deg, Gm>6dB)\n');
elseif Pm > 0
    fprintf('STABLE but marginal — consider retuning\n');
else
    fprintf('UNSTABLE — retune gains\n');
end
fprintf('═══════════════════════════════════════\n');

%% ── 9. FREQUENCY DOMAIN PLOTS ────────────────────────────────────────────
figure('Name','Frequency Domain Stability Analysis', ...
       'Position',[100 50 1000 750], 'Color','white');

subplot(2,2,[1,2]);
bodeplot(L, {0.01, 1000});
grid on;
title(sprintf('Open-Loop Bode — Pm = %.1f°,  Gm = %.1f dB', Pm, 20*log10(Gm)));

subplot(2,2,3);
nicholsplot(L);
grid on;
title('Nichols Chart — Robustness Visualization');

subplot(2,2,4);
step(CL, 5);
grid on;
title(sprintf('Closed-Loop Step Response  |  BW = %.2f Hz', bw/(2*pi)));
ylabel('Normalized Tension'); xlabel('Time (s)');

sgtitle('Adaptive Track Tension — Stability & Frequency Analysis (v3.0)', ...
        'FontSize',12,'FontWeight','bold');

figure('Name','Gain & Phase Margin Detail','Color','white');
margin(L);
grid on;
title(sprintf('Stability Margins  |  Gm = %.2f dB  |  Pm = %.2f deg', ...
              20*log10(Gm), Pm));
