% =========================================================================
% PROJECT : AI-Supported Adaptive Tension Track System
% AUTHOR  : Ayça Bahar Güner — Dokuz Eylül University, Mech. Eng.
% VERSION : 4.2 — Merged & Debugged (Monte Carlo + Stability Suite)
% DATE    : May 2026
% FIXES   : - D term uses (e-e_prev)/dt, not e/dt
%           - Anti-windup uses saturation flag, not stale-state pre-check
%           - rng('shuffle') called once before loop
%           - Derivative filter N=50 (was 0.01 — killed D entirely)
%           - results_derail re-added
% =========================================================================

clear; clc; close all;

%% ── 1. PARAMETERS (defined once, outside loop) ───────────────────────────
dt    = 0.01; t_end = 60; t = 0:dt:t_end; N = length(t);
T0    = 50000; alpha = 15; beta = 25000;
Kp    = 1.2;   Ki   = 0.10; Kd = 0.08;
tau   = 0.3;   u_max = 15000; u_min = -15000;
Nf    = 50;    % derivative filter pole [rad/s] — sensible for mech. system

%% ── 2. MONTE CARLO ANALYSIS ──────────────────────────────────────────────
num_runs       = 100;
results_rmse   = zeros(num_runs, 1);
results_derail = zeros(num_runs, 1);

fprintf('Monte Carlo analizi baslatiliyor (%d senaryo)...\n', num_runs);
rng('shuffle');  % once before loop — NOT inside

for run = 1:num_runs

    % stochastic inputs
    noise_std  = 0.5 + 0.2*rand();
    v_vehicle  = 10 + 5*sin(0.1*t)  + noise_std*randn(1,N);
    mu_terrain = 0.5 + 0.3*sin(0.05*t) + 0.05*randn(1,N);
    T_ref = max(20000, min(80000, T0 + alpha*(v_vehicle.^2) - beta*mu_terrain));

    % PID loop
    T_actual    = zeros(1,N); T_actual(1) = T0;
    e_prev = 0; e_int = 0; saturated = false;

    for i = 2:N
        e  = T_ref(i) - T_actual(i-1);
        P  = Kp * e;
        D  = Kd * (e - e_prev) / dt;   % FIXED: actual derivative

        if ~saturated                   % FIXED: integrate only when not sat.
            e_int = e_int + e*dt;
        end

        u_raw     = P + Ki*e_int + D;
        u         = max(u_min, min(u_max, u_raw));
        saturated = u_raw > u_max || u_raw < u_min;

        dT          = (u - (T_actual(i-1) - T0)) / tau;
        T_actual(i) = T_actual(i-1) + dT*dt;
        e_prev      = e;
    end

    results_rmse(run)   = sqrt(mean((T_ref - T_actual).^2));
    results_derail(run) = 100 * sum(T_actual < 20000) / N;
end

%% ── 3. STABILITY ANALYSIS ────────────────────────────────────────────────
s  = tf('s');
G  = 1 / (tau*s + 1);
C  = pid(Kp, Ki, Kd, 1/Nf);   % Tf = 1/Nf; pole at -Nf rad/s
L  = C * G;
CL = feedback(L, 1);
[Gm, Pm, Wcg, Wcp] = margin(L);

%% ── 4. CONSOLE REPORT ────────────────────────────────────────────────────
fprintf('\n═══════════════════════════════════════\n');
fprintf('  ANALIZ OZETI\n');
fprintf('═══════════════════════════════════════\n');
fprintf('  Mean RMSE     : %8.1f N\n',  mean(results_rmse));
fprintf('  RMSE Std      : %8.1f N\n',  std(results_rmse));
fprintf('  Derail Risk   : %8.2f %%\n', mean(results_derail));
fprintf('  Gain Margin   : %8.1f dB  @ %.2f rad/s\n', 20*log10(Gm), Wcg);
fprintf('  Phase Margin  : %8.1f deg @ %.2f rad/s\n', Pm, Wcp);
fprintf('═══════════════════════════════════════\n');

%% ── 5. VISUALISATION (your 2x2 layout, kept) ────────────────────────────
figure('Name','Final Integrated Report', ...
       'Position',[50 50 1200 800],'Color','white');

subplot(2,2,1);
histogram(results_rmse, 20, 'FaceColor',[0.2 0.6 0.8],'EdgeColor','white');
hold on;
xline(mean(results_rmse),'r--','LineWidth',1.5, ...
      'Label',sprintf('Mean %.0f N', mean(results_rmse)));
title('Robustness — 100-run Monte Carlo');
xlabel('RMSE [N]'); ylabel('Count'); grid on;

subplot(2,2,2);
bode(L);
title(sprintf('Open-Loop Bode | GM: %.1f dB  PM: %.1f°', 20*log10(Gm), Pm));
grid on;

subplot(2,2,3);
nicholsplot(L);
title('Nichols Chart'); grid on;

subplot(2,2,4);
step(CL, 5);
title('Closed-Loop Step Response'); grid on;
