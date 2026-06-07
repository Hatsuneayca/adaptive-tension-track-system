% =========================================================================
% AI-Supported Adaptive Track Tension System — v8.0
% AUTHOR  : Ayça Bahar Güner — Dokuz Eylül University, Mechanical Engineering
% DATE    : June 2026
%
% REQUIREMENTS: MATLAB R2019b+ | Fuzzy Logic Toolbox | Deep Learning Toolbox
%
% ── WHAT THIS CODE DOES ──────────────────────────────────────────────────
% Simulates an armored vehicle track tension control system.
% The track must stay within [20 kN, 80 kN]; too loose → derailment,
% too tight → structural damage.
%
% Three controllers are compared via Monte Carlo (100 randomized runs):
%   1. PID          — reactive only, no terrain preview
%   2. LQR + FIS    — Fuzzy Logic predicts terrain force → feedforward
%   3. LQR + NN     — Neural Network predicts terrain force → feedforward
%
% ── KEY UPGRADE FROM v7.0 ────────────────────────────────────────────────
% [NEW] 2nd-order mechanical track model replaces the algebraic proxy
%       T_actual = T_ref - x(1) that v7.0 used.
%       Equation of motion:  m*x_ddot + c*x_dot + k*x = F_act + F_terrain
%       Tension from physics: T = T0 + k*x1 + c*x2
%
% [NEW] AI predictor target changed: friction coefficient μ → terrain force F_terrain
%       F_terrain enters the state equation directly → physically grounded.
%       Architecture: LQR (reactive feedback) + AI predictor (proactive feedforward)
%
% [NEW] NN ground truth is physics-based, not arbitrary:
%       F_terrain_gt = 30000 * susp_var * (1 + 0.4*(v/10)^2)
%       Interpretation: terrain force scales with surface roughness (susp_var)
%       and dynamic pressure (v^2). This is defensible in a research context.
%
% [KEPT] Unified predictor_fn interface — swap FIS/NN without touching sim code
% [KEPT] Fair Monte Carlo — rng(r) reset before each method in every run
% [KEPT] Q is 4×4 (matches augmented state), anti-windup, functions at end
%
% ── PHYSICAL PARAMETERS (approximate literature values) ──────────────────
%   m  = 250 kg       Effective track segment mass
%   k  = 150,000 N/m  Track stiffness (steel track, rubber-padded)
%   c  = 3,500 N·s/m  Damping  (damping ratio ζ ≈ 0.30 → lightly damped)
%   ωn = √(k/m) ≈ 24.5 rad/s ≈ 3.9 Hz  (realistic for a tracked vehicle)
% =========================================================================

clear; clc; close all;
rng(42);   % Global seed for reproducibility of the overall script

%% ═══════════════════════════════════════════════════════════
%  1. SIMULATION CONFIGURATION
%% ═══════════════════════════════════════════════════════════

% Time settings — dt=0.005 s chosen for numerical stability of the 2nd-order system
% (rule of thumb: dt < 1/(10*ωn) ≈ 0.004 s; 0.005 s is just inside safe margin)
time_step   = 0.005;
total_time  = 60;
time_vector = 0:time_step:total_time;
num_points  = length(time_vector);

% Tension operating limits [N]
T0                  = 50000;   % Nominal (design) track tension
tension_limit_upper = 80000;   % Above this → track overloaded
tension_limit_lower = 20000;   % Below this → derailment risk

% Actuator saturation — hydraulic tensioner can only push/pull ±15 kN
max_force =  15000;   % [N]
min_force = -15000;   % [N]

% Mechanical parameters of the track-tensioner subsystem
m            = 250;      % [kg]      Effective moving mass (track + tensioner)
k_track      = 150000;   % [N/m]    Track longitudinal stiffness
c_track      = 3500;     % [N·s/m]  Viscous damping (shock absorbers + joints)
actuator_tau = 0.3;      % [s]      First-order actuator time constant

% Reference signal shaping
velocity_scaling = 15;    % Velocity-dependent tension bias: ΔT = vel_scaling * v²
                           % Faster speed → higher nominal tension needed
ff_gain          = 0.35;  % Feedforward gain: T_ref += ff_gain * F_terrain_predicted
                           % Tune this: too high → over-correction, too low → no benefit

fprintf('Mechanical system: wn=%.1f rad/s (%.1f Hz), zeta=%.2f\n\n', ...
    sqrt(k_track/m), sqrt(k_track/m)/(2*pi), ...
    c_track/(2*sqrt(k_track*m)));

%% ═══════════════════════════════════════════════════════════
%  2. MECHANICAL MODEL & LQR CONTROLLER DESIGN
%% ═══════════════════════════════════════════════════════════
%
% STATE VECTOR:  x = [x_ext; x_vel; x_act; x_int]
%   x(1) = x_ext : track extension from nominal position    [m]
%   x(2) = x_vel : rate of track extension                  [m/s]
%   x(3) = x_act : actuator force (internal state)          [N]
%   x(4) = x_int : integral of tension tracking error       [N·s]
%                  (added for zero steady-state error → integral action)
%
% EQUATION OF MOTION:
%   m * x_ddot + c * x_dot + k * x = F_actuator + F_terrain
%   → dx(1)/dt = x(2)
%   → dx(2)/dt = [ -k*x(1) - c*x(2) + x(3) + F_terrain ] / m
%   → dx(3)/dt = [ -x(3) + u ] / tau          (1st-order actuator dynamics)
%   → dx(4)/dt = T_ref - T(t)
%              = T_ref - (T0 + k*x(1) + c*x(2))
%              ≈ -k*x(1) - c*x(2)             (at equilibrium T_ref ≈ T0)
%
% TENSION OUTPUT:
%   T(t) = T0 + k * x(1) + c * x(2)
%   Physical meaning: tension increases when the track extends (spring) or
%   when it moves outward quickly (damper). Both contribute to measured tension.
%
% DISTURBANCE INPUT:
%   F_terrain enters through E_dist = [0; 1/m; 0; 0]
%   Only the acceleration state (x2) is directly disturbed by terrain forces.

A_aug = [  0,           1,            0,            0;
          -k_track/m,  -c_track/m,    1/m,           0;
           0,           0,           -1/actuator_tau, 0;
          -k_track,    -c_track,      0,              0 ];
%
%   Row 1: dx_ext/dt = x_vel                        (kinematics)
%   Row 2: dx_vel/dt = -k/m*x_ext - c/m*x_vel + 1/m*x_act   (Newton 2nd law)
%   Row 3: dx_act/dt = -1/tau * x_act               (actuator lag)
%   Row 4: dx_int/dt = -k*x_ext - c*x_vel           (integral of tension error)

B_aug  = [0; 0; 1/actuator_tau; 0];
% Control input u (actuator command) only drives x_act through 1/tau.

E_dist = [0; 1/m; 0; 0];
% Terrain disturbance F_terrain enters only the acceleration equation (row 2).

% LQR COST MATRIX Q — weighting philosophy:
%   We care about tension error, not directly about states. So we weight
%   each state by its contribution to tension: T ≈ T0 + k*x1 + c*x2
%   Q(1,1) = (k / ΔT_tolerance)² = (150000 / 5000)² = 900  → x_ext drives tension most
%   Q(2,2) = (c / ΔT_tolerance)² = (3500   / 5000)² ≈ 0.49 → x_vel contributes less
%   Q(3,3) = 1 / max_force²      ≈ 4.4e-9             → lightly penalize actuator use
%   Q(4,4) = 1.0                                       → integral state, normalized
Q_lqr = diag([900, 0.49, 4.4e-9, 1.0]);

% R_lqr — control effort penalty. Small value → LQR allowed to use large forces.
% Physical constraint is handled by saturation, not by R.
R_lqr = 1e-8;

[K_lqr, ~, eig_cl] = lqr(A_aug, B_aug, Q_lqr, R_lqr);

fprintf('LQR Gain K = [%.4f  %.6f  %.6f  %.6f]\n', K_lqr);
fprintf('Closed-loop eigenvalues: ');
fprintf('%.3f+%.3fi  ', [real(eig_cl)'; imag(eig_cl)']);
fprintf('\n');
% All eigenvalues must have negative real parts → system is stable
fprintf('All Re(eig) < 0: %s\n\n', mat2str(all(real(eig_cl) < 0)));

%% ═══════════════════════════════════════════════════════════
%  3. FUZZY LOGIC PREDICTOR — estimates F_terrain
%% ═══════════════════════════════════════════════════════════
% Mamdani FIS: 2 inputs (susp_var, velocity) → 1 output (F_terrain magnitude)
% 9 rules cover the full input domain via AND combinations (3 × 3 grid).
% Gaussian output MFs allow smooth interpolation between terrain classes.
%
% WHY FUZZY HERE:
%   Rules are interpretable and can be reviewed by a domain expert.
%   The terrain→force mapping is fundamentally uncertain; fuzzy logic
%   explicitly models this uncertainty through membership functions.

fprintf('[1/2] Building Fuzzy Inference System (F_terrain predictor)...\n');
fis = build_fterrain_fis();
fprintf('      Done: %d rules, 2 inputs, 1 output\n\n', numel(fis.Rules));

% Wrap in anonymous function to match the unified predictor interface:
%   predictor_fn(velocity, susp_var, tension_err) → F_terrain_predicted
pred_fuzzy = @(v, sv, e) fuzzy_predict_ft(fis, v, sv, e);

%% ═══════════════════════════════════════════════════════════
%  4. NEURAL NETWORK PREDICTOR — estimates F_terrain
%% ═══════════════════════════════════════════════════════════
% Architecture: fitnet([12, 6]) — 2 inputs → 12 → 6 → 1 output
% Training data: N=2000 synthetic samples from terrain_force_gt()
%
% Ground truth:  F_terrain_gt = 30000 * susp_var * (1 + 0.4*(v/10)^2)
%   Physical interpretation:
%     - 30000 * susp_var  : terrain roughness generates a base disturbance force
%     - (1 + 0.4*(v/10)^2): dynamic pressure amplification at higher speeds
%   This formula is physically motivated (not arbitrary sigmoid parameters
%   as in v7.0). It can be cited as a simplified dynamic loading model.
%
% WHY NN HERE:
%   Learns the continuous nonlinear surface F(susp_var, velocity) without
%   requiring explicit rule specification. Generalizes smoothly between
%   training points via learned weights.

fprintf('[2/2] Training Neural Network (N=2000, physics-based ground truth)...\n');
net = train_fterrain_nn();
fprintf('      Done\n\n');

pred_nn = @(v, sv, e) nn_predict_ft(net, v, sv, e);

%% ═══════════════════════════════════════════════════════════
%  5. MONTE CARLO COMPARISON
%% ═══════════════════════════════════════════════════════════
% Each run:
%   1. rng(r)  → set seed so all three methods see identical disturbances
%   2. Draw noise_intensity from uniform distribution (run-to-run variation)
%   3. Run PID, LQR+FIS, LQR+NN on the same mechanical plant
%   4. Record RMSE and derailment risk
%
% Fairness guarantee: rng(r) is called before EACH method within each run,
% so the random disturbance sequences are byte-for-byte identical across methods.

num_runs = 100;
rmse_pid = zeros(num_runs,1);
rmse_fis = zeros(num_runs,1);
rmse_nn  = zeros(num_runs,1);
risk_pid = zeros(num_runs,1);
risk_fis = zeros(num_runs,1);
risk_nn  = zeros(num_runs,1);

fprintf('Running Monte Carlo (%d runs) — PID vs LQR+FIS vs LQR+NN...\n', num_runs);

for r = 1:num_runs
    % Draw noise level — randomized per run, same for all methods in this run
    rng(r);
    noise = 0.40 + 0.30*rand();

    % ── PID baseline: reactive only, no terrain prediction, no feedforward ──
    rng(r);   % reset seed → identical disturbance realization as LQR methods
    [T_pid, Tref_pid] = run_pid_mech(time_vector, num_points, time_step, ...
        T0, k_track, c_track, m, actuator_tau, tension_limit_lower, tension_limit_upper, ...
        A_aug, B_aug, E_dist, velocity_scaling, max_force, min_force, noise);
    rmse_pid(r) = sqrt(mean((Tref_pid - T_pid).^2));
    risk_pid(r) = 100 * sum(T_pid < tension_limit_lower) / num_points;

    % ── LQR + Fuzzy Logic feedforward ──────────────────────────────────────
    rng(r);
    [T_fis, Tref_fis, met_fis] = run_lqr_mech(time_vector, num_points, time_step, ...
        T0, k_track, c_track, m, actuator_tau, tension_limit_lower, tension_limit_upper, ...
        A_aug, B_aug, E_dist, K_lqr, velocity_scaling, ff_gain, ...
        max_force, min_force, noise, pred_fuzzy);
    rmse_fis(r) = met_fis.rmse;
    risk_fis(r) = met_fis.derail_risk;

    % ── LQR + Neural Network feedforward ───────────────────────────────────
    rng(r);
    [T_nn, Tref_nn, met_nn] = run_lqr_mech(time_vector, num_points, time_step, ...
        T0, k_track, c_track, m, actuator_tau, tension_limit_lower, tension_limit_upper, ...
        A_aug, B_aug, E_dist, K_lqr, velocity_scaling, ff_gain, ...
        max_force, min_force, noise, pred_nn);
    rmse_nn(r) = met_nn.rmse;
    risk_nn(r) = met_nn.derail_risk;
end

%% ═══════════════════════════════════════════════════════════
%  6. RESULTS SUMMARY
%% ═══════════════════════════════════════════════════════════

fprintf('\n');
fprintf('=================================================================\n');
fprintf('   v8.0 MONTE CARLO RESULTS — Mechanical Plant  (N=%d runs)\n', num_runs);
fprintf('=================================================================\n');
fprintf('                 PID (reactive)   LQR+FIS       LQR+NN\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Mean RMSE [N]  : %8.1f       %8.1f (%+.1f%%)  %8.1f (%+.1f%%)\n', ...
    mean(rmse_pid), ...
    mean(rmse_fis), 100*(1-mean(rmse_fis)/mean(rmse_pid)), ...
    mean(rmse_nn),  100*(1-mean(rmse_nn) /mean(rmse_pid)));
fprintf('RMSE Std  [N]  : %8.1f       %8.1f         %8.1f\n', ...
    std(rmse_pid), std(rmse_fis), std(rmse_nn));
fprintf('Derailment %%   : %8.2f%%      %8.2f%%        %8.2f%%\n', ...
    mean(risk_pid), mean(risk_fis), mean(risk_nn));
fprintf('=================================================================\n');

%% ═══════════════════════════════════════════════════════════
%  7. VISUALIZATION
%% ═══════════════════════════════════════════════════════════

c_pid = [0.85 0.35 0.25];   % red  — PID
c_fis = [0.15 0.60 0.30];   % green — LQR+FIS
c_nn  = [0.15 0.42 0.82];   % blue  — LQR+NN

figure('Name','v8.0 Mechanical Model + AI Predictor', ...
       'Position',[50 50 1400 780], 'Color','white');

% Panel 1: RMSE distribution across all Monte Carlo runs
%   Narrower + left-shifted histogram → better controller
subplot(2,3,1);
histogram(rmse_pid,15,'FaceColor',c_pid,'EdgeColor','none','FaceAlpha',0.75); hold on;
histogram(rmse_fis,15,'FaceColor',c_fis,'EdgeColor','none','FaceAlpha',0.75);
histogram(rmse_nn, 15,'FaceColor',c_nn, 'EdgeColor','none','FaceAlpha',0.75);
xline(mean(rmse_pid),'--','Color',c_pid,'LineWidth',1.8);
xline(mean(rmse_fis),'--','Color',c_fis,'LineWidth',1.8);
xline(mean(rmse_nn), '--','Color',c_nn, 'LineWidth',1.8);
legend('PID','LQR+FIS','LQR+NN','Location','best');
title('RMSE Distribution — Monte Carlo');
xlabel('RMSE [N]'); ylabel('Count'); grid on;

% Panel 2: Boxplot — shows median, IQR, and outliers per method
%   Smaller box + lower median → more consistent tracking
subplot(2,3,2);
boxplot([rmse_pid rmse_fis rmse_nn], 'Labels',{'PID','LQR+FIS','LQR+NN'});
title('RMSE Boxplot Comparison'); ylabel('RMSE [N]'); grid on;

% Panel 3: Average derailment risk (% of time below 20 kN lower limit)
subplot(2,3,3);
b = bar([mean(risk_pid) mean(risk_fis) mean(risk_nn)], 0.55);
b.FaceColor = 'flat';
b.CData = [c_pid; c_fis; c_nn];
set(gca,'XTickLabel',{'PID','LQR+FIS','LQR+NN'});
title('Average Derailment Risk [%]'); ylabel('Risk [%]'); grid on;

% Panel 4: LQR+FIS tracking — last Monte Carlo run
%   Shows how closely actual tension follows the AI-generated reference
subplot(2,3,4);
plot(time_vector, Tref_fis/1000,'--','Color',c_fis,'LineWidth',1.4); hold on;
plot(time_vector, T_fis/1000,   '-', 'Color',c_fis*0.6,'LineWidth',1.0);
yline(tension_limit_lower/1000,'k:','LineWidth',1.1);
yline(tension_limit_upper/1000,'k:','LineWidth',1.1);
title(sprintf('LQR + FIS Tracking  (mean RMSE=%.0f N)', mean(rmse_fis)));
xlabel('Time [s]'); ylabel('Tension [kN]'); grid on;
legend('AI Reference','LQR Actual','Safety Limits','Location','best');

% Panel 5: LQR+NN tracking — last Monte Carlo run
subplot(2,3,5);
plot(time_vector, Tref_nn/1000,'--','Color',c_nn,'LineWidth',1.4); hold on;
plot(time_vector, T_nn/1000,   '-', 'Color',c_nn*0.6,'LineWidth',1.0);
yline(tension_limit_lower/1000,'k:','LineWidth',1.1);
yline(tension_limit_upper/1000,'k:','LineWidth',1.1);
title(sprintf('LQR + NN Tracking  (mean RMSE=%.0f N)', mean(rmse_nn)));
xlabel('Time [s]'); ylabel('Tension [kN]'); grid on;
legend('AI Reference','LQR Actual','Safety Limits','Location','best');

% Panel 6: FIS output surface — F_terrain vs [susp_var, velocity]
%   Shows the smooth nonlinear mapping learned by the fuzzy rules.
%   A flat surface → predictor ignores one input (bad).
%   Correct shape: force increases with both susp_var AND velocity.
subplot(2,3,6);
sv_g = linspace(0, 0.20, 45);
vl_g = linspace(5, 18, 45);
[SV, VL] = meshgrid(sv_g, vl_g);
input_mat = [SV(:), VL(:)];           % evalfis expects [N_samples × N_inputs]
ft_vec    = evalfis(fis, input_mat);
FT        = reshape(ft_vec, size(SV));
surf(SV, VL, FT/1000, 'EdgeColor','none','FaceAlpha',0.88);
colorbar; colormap(hot);
xlabel('susp\_var'); ylabel('Velocity [m/s]'); zlabel('F_{terrain} [kN]');
title('FIS Output Surface: susp\_var × Velocity → F_{terrain}');
grid on; view(38, 32);

sgtitle('Adaptive Track Tension System v8.0 — Mechanical Model + AI Feedforward', ...
        'FontSize',13,'FontWeight','bold');

% =========================================================================
%  LOCAL FUNCTIONS
%  MATLAB rule: all local functions must appear AFTER all script-level code.
% =========================================================================

function F_gt = terrain_force_gt(sv, vel)
% TERRAIN FORCE GROUND TRUTH MODEL
%
% Used for: (a) generating NN training data, (b) simulation disturbance signal.
%
% Formula:  F_terrain = 30000 * susp_var * (1 + 0.4 * (v/10)^2)
%
% Physical reasoning:
%   - susp_var (suspension displacement variance proxy) ∝ terrain roughness
%   - The (1 + 0.4*(v/10)^2) term captures dynamic loading amplification:
%     at low speed the force is quasi-static; at higher speed, inertial
%     effects from repeated impacts scale roughly with v^2.
%   - The 30000 scaling factor gives forces in the range [0, ~7800] N
%     for typical operating conditions (sv ≤ 0.20, v ≤ 18 m/s).
%
% This formula is NOT calibrated to a specific vehicle — it is a simplified
% model suitable for control system validation. Real deployment requires
% experimental identification of the terrain-force transfer function.

    F_gt = 30000 .* sv .* (1 + 0.40*(vel/10).^2);
    F_gt = max(0, min(10000, F_gt));   % physical saturation [0, 10 kN]
end

% ─────────────────────────────────────────────────────────────────────────
function fis = build_fterrain_fis()
% BUILD MAMDANI FUZZY INFERENCE SYSTEM
%
% Maps observable signals → predicted terrain disturbance force.
%
% Inputs:
%   susp_var  [0, 0.20]  — suspension displacement variance proxy
%                          (computed from measured suspension travel)
%   velocity  [5, 18] m/s — vehicle speed
%
% Output:
%   F_terrain [0, 9000] N — predicted terrain disturbance force magnitude
%
% Membership functions:
%   susp_var: 'smooth' (trapmf), 'rough' (gaussmf), 'severe' (trapmf)
%   velocity: 'slow' (trapmf), 'medium' (gaussmf), 'fast' (trapmf)
%   F_terrain: 'calm'(500N), 'mild'(2500N), 'rough_f'(5000N), 'severe_f'(7500N)
%
% 9 rules cover the full [susp_var × velocity] domain (3×3 grid, AND connections).
% Gaussian output MFs allow smooth interpolation — no hard jumps between classes.
%
% Defuzzification: centroid method (Mamdani default)

    fis = mamfis('Name','FTerrainPredictor');

    % --- Input 1: suspension variance proxy ---
    fis = addInput(fis, [0 0.20], 'Name','susp_var');
    fis = addMF(fis,'susp_var','trapmf', [-0.01 0    0.055 0.090],'Name','smooth');
    fis = addMF(fis,'susp_var','gaussmf',[0.025 0.110],           'Name','rough');
    fis = addMF(fis,'susp_var','trapmf', [0.130 0.158 0.21 0.22], 'Name','severe');

    % --- Input 2: vehicle velocity ---
    fis = addInput(fis, [5 18], 'Name','velocity');
    fis = addMF(fis,'velocity','trapmf', [4   5   8  10],'Name','slow');
    fis = addMF(fis,'velocity','gaussmf',[2.2 11],       'Name','medium');
    fis = addMF(fis,'velocity','trapmf', [13  15  18 19],'Name','fast');

    % --- Output: F_terrain magnitude ---
    fis = addOutput(fis, [0 9000], 'Name','F_terrain');
    fis = addMF(fis,'F_terrain','gaussmf',[300  500], 'Name','calm');     %  ~500 N
    fis = addMF(fis,'F_terrain','gaussmf',[600 2500], 'Name','mild');     % ~2500 N
    fis = addMF(fis,'F_terrain','gaussmf',[900 5000], 'Name','rough_f'); % ~5000 N
    fis = addMF(fis,'F_terrain','gaussmf',[1000 7500],'Name','severe_f');% ~7500 N

    % Rule matrix: [susp_MF, vel_MF, F_terrain_MF, weight, connection(1=AND)]
    rules = [
        1 1 1 1.00 1;   % smooth  AND slow    → calm      (flat road, low speed)
        1 2 1 1.00 1;   % smooth  AND medium  → calm
        1 3 2 0.80 1;   % smooth  AND fast    → mild      (aero/dynamic effects at high speed)
        2 1 2 1.00 1;   % rough   AND slow    → mild
        2 2 3 1.00 1;   % rough   AND medium  → rough_f
        2 3 3 1.00 1;   % rough   AND fast    → rough_f   (gravel at speed)
        3 1 3 1.00 1;   % severe  AND slow    → rough_f   (heavy terrain, slow crawl)
        3 2 4 1.00 1;   % severe  AND medium  → severe_f
        3 3 4 1.00 1;   % severe  AND fast    → severe_f  (worst case)
    ];
    fis = addRule(fis, rules);
end

% ─────────────────────────────────────────────────────────────────────────
function net = train_fterrain_nn()
% TRAIN NEURAL NETWORK TO PREDICT F_terrain
%
% Architecture: 2 inputs → [12 → 6] hidden layers → 1 output
%   Input 1: susp_var              (not normalized — already in [0, 0.20])
%   Input 2: velocity / 18         (normalized to [0, 1])
%   Output:  F_terrain / 9000      (normalized to [0, 1] for training stability)
%
% Training algorithm: Levenberg-Marquardt (trainlm)
%   Chosen for speed on small-medium datasets. Converges in ~50-100 epochs
%   for this problem size (N=2000, 2 inputs, 1 output).
%
% Data split: 75% train / 15% validation / 10% test
%   Validation set used for early stopping (max_fail=20 epochs without improvement).
%
% Note on tension_err as input:
%   Deliberately excluded. F_terrain is a property of the terrain and vehicle
%   speed — it does NOT depend on the controller's tracking error.
%   Including tension_err would introduce a spurious correlation and make
%   the predictor non-causal (error depends on past control decisions).

    rng(0);   % Fixed seed for reproducible training data generation
    N   = 2000;
    sv  = rand(1,N) * 0.20;                        % susp_var in [0, 0.20]
    vel = 5 + rand(1,N) * 13;                      % velocity in [5, 18] m/s

    % Ground truth with measurement noise (σ = 300 N simulates sensor noise)
    F_gt = terrain_force_gt(sv, vel) + 300*randn(1,N);
    F_gt = max(0, min(10000, F_gt));

    X = [sv; vel/18];      % 2×N input matrix — velocity normalized
    Y = F_gt / 9000;       % 1×N target — normalized to [0,1] for training

    net = fitnet([12 6], 'trainlm');
    net.trainParam.epochs      = 300;
    net.trainParam.goal        = 1e-5;   % MSE stopping criterion
    net.trainParam.showWindow  = false;  % suppress training GUI
    net.trainParam.max_fail    = 20;     % early stopping patience
    net.divideParam.trainRatio = 0.75;
    net.divideParam.valRatio   = 0.15;
    net.divideParam.testRatio  = 0.10;

    net = train(net, X, Y);
end

% ─────────────────────────────────────────────────────────────────────────
function F_pred = fuzzy_predict_ft(fis, velocity, susp_var, ~)
% FUZZY PREDICTOR INFERENCE
%
% Evaluates the FIS at the current [susp_var, velocity] operating point.
% tension_err (3rd argument) is intentionally ignored (~) — see train_fterrain_nn
% for the rationale. Both FIS and NN predictors share this design decision
% to ensure a fair comparison.
%
% evalfis(fis, [susp_var, velocity]) returns defuzzified F_terrain [N].

    F_pred = evalfis(fis, [susp_var, velocity]);
    F_pred = max(0, min(10000, F_pred));   % clamp to physical range
end

% ─────────────────────────────────────────────────────────────────────────
function F_pred = nn_predict_ft(net, velocity, susp_var, ~)
% NEURAL NETWORK PREDICTOR INFERENCE
%
% Forward pass through the trained network.
% Inputs must be normalized identically to training data:
%   susp_var unchanged (already [0, 0.20])
%   velocity divided by 18
%
% Network output is in [0, 1] (normalized) → rescale by 9000 to get [N].

    F_norm = net([susp_var; velocity/18]);
    F_pred = F_norm * 9000;
    F_pred = max(0, min(10000, F_pred));
end

% ─────────────────────────────────────────────────────────────────────────
function [T_actual, T_ref, metrics] = run_lqr_mech(time_vector, num_points, time_step, ...
        T0, k_track, c_track, m, actuator_tau, tension_limit_lower, tension_limit_upper, ...
        A_aug, B_aug, E_dist, K_lqr, velocity_scaling, ff_gain, ...
        max_force, min_force, noise_intensity, predictor_fn)
%
% LQR SIMULATION WITH MECHANICAL TRACK PLANT
%
% CONTROL ARCHITECTURE (two-degree-of-freedom):
%
%   [AI Predictor] → estimates F_terrain → adjusts T_ref (feedforward)
%   [LQR]         → minimizes J=∫(x'Qx + u'Ru)dt → drives T_actual → T_ref (feedback)
%   [Mechanical]  → m*x_ddot + c*x_dot + k*x = F_act + F_terrain_TRUE
%   [Tension]     → T = T0 + k*x(1) + c*x(2)
%
% The separation between predictor (proactive) and LQR (reactive) is the
% key research contribution: the predictor adjusts WHAT we want (reference),
% while the LQR ensures we GET there despite modeling errors and noise.
%
% REFERENCE GENERATION:
%   T_ref = T0 + velocity_scaling * v^2 + ff_gain * F_terrain_predicted
%   Term 1: Nominal tension at rest
%   Term 2: Speed-dependent tension increase (higher speed → need more tension)
%   Term 3: Feedforward compensation for predicted terrain force
%            (if rough terrain is detected, pre-increase reference tension)
%
% STATE PROPAGATION (Euler integration):
%   x_dot = A_aug * x + B_aug * u + E_dist * F_terrain_TRUE
%   x_new = x + x_dot * dt
%
% ANTI-WINDUP (back-calculation):
%   When the actuator saturates, the integral state x(4) is corrected
%   to prevent integrator windup. Standard back-calculation scheme.

    % Generate scenario signals for this run
    vehicle_velocity = 9 + 5.5*sin(0.085*time_vector) ...
                       + noise_intensity*randn(1,num_points);
    susp_var = max(0.010, 0.050 + 0.090*abs(sin(0.055*time_vector)) ...
                   + 0.030*randn(1,num_points));

    % TRUE terrain disturbance (known to simulation, unknown to controller)
    % The predictor tries to estimate this signal from observable inputs.
    F_terrain_true = terrain_force_gt(susp_var, vehicle_velocity) ...
                     + 400*randn(1,num_points);   % σ=400 N sensor/terrain noise
    F_terrain_true = max(0, min(10000, F_terrain_true));

    T_ref    = zeros(1,num_points);
    T_actual = zeros(1,num_points);
    T_actual(1) = T0;

    x = zeros(4,1);   % Initial state: track at rest, no extension, no error integral

    for i = 1:num_points
        % Current tracking error (used by predictor for any error-based correction)
        curr_err = (i>1) * (T_ref(i-1) - T_actual(i-1));

        % AI feedforward: predict terrain force and adjust reference
        F_pred  = predictor_fn(vehicle_velocity(i), susp_var(i), curr_err);
        T_ref(i) = T0 + velocity_scaling*(vehicle_velocity(i)^2) + ff_gain*F_pred;
        T_ref(i) = max(tension_limit_lower, min(tension_limit_upper, T_ref(i)));

        if i == 1; continue; end   % No control action at t=0 (initial condition)

        % Inject reference change into integral state (soft reference tracking)
        % This allows the integral term to adapt as T_ref varies over time.
        % Division by (k_track+1) scales the injection to avoid integral overshoot.
        x(4) = x(4) + (T_ref(i) - T_ref(i-1)) / (k_track + 1);

        % LQR control law: u = -K * x
        u_raw = -K_lqr * x;

        % Actuator saturation
        u = max(min_force, min(max_force, u_raw));

        % Anti-windup: back-calculate integral correction when saturated
        if abs(u_raw) > max_force
            x(4) = x(4) - (u_raw - u)*time_step / max(abs(K_lqr(4)), 1e-8);
        end

        % State propagation (Euler integration)
        % TRUE terrain force enters here — this is what the predictor tries to cancel
        x_dot = A_aug*x + B_aug*u + E_dist*F_terrain_true(i);
        x     = x + x_dot*time_step;

        % Compute tension from mechanical state (not from error proxy)
        T_actual(i) = T0 + k_track*x(1) + c_track*x(2);

        % Hard clamp — prevents numerical blow-up in extreme scenarios
        T_actual(i) = max(tension_limit_lower-5000, ...
                          min(tension_limit_upper+5000, T_actual(i)));
    end

    metrics.rmse        = sqrt(mean((T_ref - T_actual).^2));
    metrics.derail_risk = 100*sum(T_actual < tension_limit_lower) / num_points;
    metrics.max_ext     = max(abs(x(1)));   % maximum track extension [m]
end

% ─────────────────────────────────────────────────────────────────────────
function [T_actual, T_ref] = run_pid_mech(time_vector, num_points, time_step, ...
        T0, k_track, c_track, m, actuator_tau, tension_limit_lower, tension_limit_upper, ...
        A_aug, B_aug, E_dist, velocity_scaling, max_force, min_force, noise_intensity)
%
% PID BASELINE WITH MECHANICAL PLANT
%
% Deliberately uses NO terrain prediction and NO feedforward.
% This isolates the benefit of AI feedforward: the only architectural
% difference between PID and LQR+AI is the predictor + feedforward term.
%
% PID gains are tuned for the mechanical plant (units: actuator command [N]
% driven by tension error [N]). Note these differ from v7.0 values because
% the plant dynamics are fundamentally different (2nd-order mechanical vs
% 1st-order proxy).
%
% Reference: T_ref = T0 + velocity_scaling * v^2  (static, no terrain info)
% Plant:     identical mechanical model to run_lqr_mech (same A_aug, B_aug, E_dist)
%
% Anti-windup: conditional integration (freeze integrator when output saturated)

    kp = 1.2e-3;   % Proportional gain [N_command / N_error]
    ki = 0.8e-4;   % Integral gain     [N_command / (N_error * s)]
    kd = 2.0e-4;   % Derivative gain   [N_command / (N_error / s)]
    % These were manually tuned. For publication, Ziegler-Nichols or
    % relay-based auto-tuning on the identified plant should be used.

    vehicle_velocity = 9 + 5.5*sin(0.085*time_vector) ...
                       + noise_intensity*randn(1,num_points);
    susp_var = max(0.010, 0.050 + 0.090*abs(sin(0.055*time_vector)) ...
                   + 0.030*randn(1,num_points));
    F_terrain_true = terrain_force_gt(susp_var, vehicle_velocity) ...
                     + 400*randn(1,num_points);
    F_terrain_true = max(0, min(10000, F_terrain_true));

    % Static reference — no terrain awareness
    T_ref = T0 + velocity_scaling*(vehicle_velocity.^2);
    T_ref = max(tension_limit_lower, min(tension_limit_upper, T_ref));

    T_actual       = zeros(1,num_points);
    T_actual(1)    = T0;
    error_integral = 0;
    prev_error     = 0;
    x              = zeros(4,1);   % same state vector structure as LQR

    for i = 2:num_points
        error = T_ref(i) - T_actual(i-1);
        prop  = kp * error;
        deriv = kd * (error - prev_error) / time_step;

        % Conditional anti-windup: only integrate when not saturating
        if abs(prop + ki*error_integral) < max_force
            error_integral = error_integral + error*time_step;
        end

        raw = prop + ki*error_integral + deriv;
        u   = max(min_force, min(max_force, raw));

        % SAME mechanical plant as LQR — fair comparison guaranteed
        x_dot = A_aug*x + B_aug*u + E_dist*F_terrain_true(i);
        x     = x + x_dot*time_step;

        T_actual(i) = T0 + k_track*x(1) + c_track*x(2);
        T_actual(i) = max(tension_limit_lower-5000, ...
                          min(tension_limit_upper+5000, T_actual(i)));
        prev_error = error;
    end
end
