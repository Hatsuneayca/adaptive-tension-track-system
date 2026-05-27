% =========================================================================
% BODE & STABILITY ANALYSIS — Adaptive Track Tension System
% Append this section to adaptive_tension_control.m OR run standalone
% (ensure Kp, Ki, Kd, tau are defined in workspace first)
% =========================================================================

%% ── 8. FREQUENCY DOMAIN — STABILITY ANALYSIS ────────────────────────────

% --- Transfer Functions ---
s = tf('s');

% Plant: first-order hydraulic lag
G = 1 / (tau*s + 1);                    % [N/N], tau = 0.3s

% PID controller (parallel form)
% C(s) = Kp + Ki/s + Kd*s
% Note: pure derivative Kd*s is improper — adding low-pass filter (N=100)
N_filter = 100;                          % derivative filter coefficient
C = pid(Kp, Ki, Kd, 1/N_filter);        % proper PID with derivative filter

% Open-loop and closed-loop
L  = C * G;                              % open-loop  L(s) = C(s)·G(s)
CL = feedback(L, 1);                     % closed-loop T(s) = L/(1+L)

% --- Gain & Phase Margins ---
[Gm, Pm, Wcg, Wcp] = margin(L);

fprintf('\n═══════════════════════════════════════\n');
fprintf('  STABILITY MARGINS\n');
fprintf('═══════════════════════════════════════\n');
fprintf('  Gain margin          : %6.2f dB   (at %.2f rad/s)\n', 20*log10(Gm), Wcg);
fprintf('  Phase margin         : %6.2f deg  (at %.2f rad/s)\n', Pm, Wcp);
fprintf('  Stability assessment : ');
if Pm > 45 && 20*log10(Gm) > 6
    fprintf('ROBUST (Pm>45°, Gm>6dB)\n');
elseif Pm > 0
    fprintf('STABLE but marginal — consider retuning\n');
else
    fprintf('UNSTABLE — retune gains\n');
end
fprintf('═══════════════════════════════════════\n\n');

% --- Bandwidth & Rise Time Estimate ---
bw = bandwidth(CL);
fprintf('  Closed-loop bandwidth : %.2f rad/s (≈ %.2f Hz)\n\n', bw, bw/(2*pi));

%% ── 9. PLOTS ─────────────────────────────────────────────────────────────

figure('Name','Frequency Domain Stability Analysis', ...
       'Position',[100 50 1000 750], 'Color','white');

% Subplot 1: Bode of open-loop L(s)
subplot(2,2,[1,2]);
bodeplot(L, {0.01, 1000});
grid on;
title(sprintf('Open-Loop Bode — L(s) = C(s)·G(s)  |  Pm = %.1f°,  Gm = %.1f dB', ...
              Pm, 20*log10(Gm)));

% Subplot 2: Nichols chart (gain/phase plane — useful for robustness vis.)
subplot(2,2,3);
nicholsplot(L);
grid on;
title('Nichols Chart — Robustness Visualization');

% Subplot 3: Closed-loop step response
subplot(2,2,4);
step(CL, 5);            % 5 second window
grid on;
title(sprintf('Closed-Loop Step Response  |  BW = %.2f Hz', bw/(2*pi)));
ylabel('Normalized Tension');
xlabel('Time (s)');

sgtitle('Adaptive Track Tension — Stability & Frequency Analysis', ...
        'FontSize',12,'FontWeight','bold');

%% ── 10. MARGIN PLOT (separate figure — classic presentation) ─────────────
figure('Name','Gain & Phase Margin Detail','Color','white');
margin(L);
grid on;
title(sprintf('Stability Margins  |  Gm = %.2f dB  |  Pm = %.2f°', ...
              20*log10(Gm), Pm));
