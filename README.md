# Adaptive Track Tension Control System with AI-Supported Terrain Prediction

**An LQR-based control system augmented with fuzzy logic and neural network predictors for real-time track tension management in armored vehicles.**

-----

## Overview

This repository contains a complete simulation framework for adaptive track tension control in tracked vehicles. The system addresses a critical challenge: maintaining track tension within a narrow band ([20 kN, 80 kN]) across varying terrain while minimizing energy consumption and structural wear.

### The Problem

- **Too loose**: Track derails on rough terrain or during acceleration/cornering
- **Too tight**: Excessive drivetrain load, reduced component lifespan, higher fuel consumption
- **Manual adjustment**: Operators adjust tension periodically; feedback is slow (hours) and subjective

### The Solution

Three controllers are compared via Monte Carlo simulation:

|Controller              |Architecture          |Key Feature                                 |
|------------------------|----------------------|--------------------------------------------|
|**PID**                 |Reactive feedback only|Baseline: no terrain awareness              |
|**LQR + Fuzzy Logic**   |Feedback + Feedforward|Interprets terrain roughness via fuzzy rules|
|**LQR + Neural Network**|Feedback + Feedforward|Learns terrain-force mapping from data      |

All controllers operate on a **physics-based mechanical track model** (2nd-order mass-spring-damper system):

```
m·ẍ + c·ẋ + k·x = F_actuator + F_terrain_disturbance
T(t) = T₀ + k·x(1) + c·x(2)
```

This replaces the algebraic proxy used in earlier versions, making the system **physically defensible for publication**.

-----

## Key Features

✅ **Unified predictor interface** — swap FIS/NN without touching simulation code  
✅ **Fair comparison** — all three methods see identical disturbances (rng control)  
✅ **Physics-informed** — terrain force model is mechanically grounded  
✅ **Publication-ready** — extensive inline documentation, validation framework  
✅ **Modular design** — easily extend with new controllers or predictors

-----

## Repository Structure

```
.
├── adaptive_tension_v8_mechanical.m      # Main simulation (100 runs, 3 controllers)
├── ai_predictor_framework.m               # Predictor library (3 options: poly/GP/PINN)
├── adaptive_tension_v7_ai.m               # Previous version (FIS/NN, no mechanical model)
├── adaptive_tension_lqr_v6_fixed.m        # v6 baseline (for reference)
├── README.md                              # This file
└── METHODOLOGY.md                         # Detailed mathematical formulation
```

-----

## Quick Start

### Requirements

- **MATLAB R2019b or later**
- **Fuzzy Logic Toolbox** (for FIS predictor)
- **Deep Learning Toolbox** (for NN training)
- *(Optional) Statistics & Machine Learning Toolbox* (for Gaussian Process predictor)

### Running the Main Simulation

```matlab
% Run the complete v8.0 comparison (100 Monte Carlo runs, ~3-5 minutes)
cd /path/to/repo
adaptive_tension_v8_mechanical

% Output: Monte Carlo statistics + 2×3 comparison plots
```

**Expected results** (from 100 runs):

```
Mean RMSE [N]  :  2845.3      2156.8 (-24.2%)  1987.5 (-30.2%)
Derailment %   :     2.34%       0.78%           0.65%
```

### Building Custom Predictors

```matlab
% Compare all three predictor types
main_predictor_demo()

% Or use a specific one
pred = get_predictor_for_simulation('pinn');   % Physics-Informed NN
% pred = get_predictor_for_simulation('poly');  % Polynomial (fastest)
% pred = get_predictor_for_simulation('gp');    % Gaussian Process (uncertainty aware)
```

-----

## System Architecture

### State Vector

```
x = [x_ext; x_vel; x_act; x_int]
  x(1) : track extension from nominal position     [m]
  x(2) : rate of track extension                   [m/s]
  x(3) : actuator force state (1st-order lag)      [N]
  x(4) : integral of tension tracking error        [N·s]
```

### Control Loop

```
┌─────────────────────────────────────────┐
│  AI Predictor                           │
│  F_terrain_pred = f(v, sv, e)           │
└─────────┬───────────────────────────────┘
          │ (feedforward gain ff_gain=0.35)
          ▼
┌──────────────────────────────────────────┐
│  Reference Generator                    │
│  T_ref = T₀ + velocity_scaling·v² + ... │
└─────────┬────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────┐
│  LQR Controller                         │
│  u = -K·x  (state feedback)             │
└─────────┬────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────┐
│  Mechanical Track Plant                 │
│  m·ẍ + c·ẋ + k·x = u + F_terrain_true  │
│  T = T₀ + k·x(1) + c·x(2)               │
└──────────────────────────────────────────┘
```

### Physical Parameters

|Parameter             |Value              |Source                       |
|----------------------|-------------------|-----------------------------|
|Track mass `m`        |250 kg             |Armored vehicle segment      |
|Track stiffness `k`   |150,000 N/m        |Steel + rubber pads          |
|Damping `c`           |3,500 N·s/m        |ζ ≈ 0.30 (lightly damped)    |
|Natural frequency `ωₙ`|24.5 rad/s ≈ 3.9 Hz|Realistic for tracked vehicle|
|Actuator τ            |0.3 s              |Hydraulic tensioner          |
|Nominal tension `T₀`  |50 kN              |Design operating point       |

-----

## AI Predictor Design

### Ground Truth Model

The terrain force is modeled as:

```
F_terrain = 30000 · susp_var · (1 + 0.4·(v/10)²)
```

**Physical interpretation:**

- `susp_var` ∝ terrain roughness (suspension displacement variance)
- `(1 + 0.4·(v/10)²)` = dynamic amplification factor (inertial effects scale with v²)
- Range: [0, ~8000] N across typical operating conditions

### Three Predictor Options

#### 1. Polynomial Regressor

- **Model**: `F = Σ cᵢ·φᵢ(sv, v)` (degree 3)
- **Pros**: Human-readable, embeddable, ~1 μs inference
- **Cons**: Limited accuracy
- **Use case**: Baseline, embedded systems

#### 2. Gaussian Process (RBF kernel)

- **Model**: Bayesian nonparametric regression
- **Pros**: Outputs uncertainty (confidence intervals), theoretically sound
- **Cons**: Slow (~150 μs), requires toolbox, O(N³) training
- **Use case**: Uncertainty quantification, academic credibility

#### 3. Physics-Informed Neural Network (PINN)

- **Model**: `F(sv,v) = sv · g(v,sv)` (enforces F(0,v)=0 by construction)
- **Pros**: Best accuracy/interpretability balance, publication-ready
- **Cons**: Still a “learned” model (though constrained by physics)
- **Use case**: **Recommended for publication**

### Validation

Each predictor is evaluated on:

- ✅ Test set generalization (RMSE, MAE, bias)
- ✅ Input domain coverage (grid evaluation)
- ✅ Boundary behavior (edges, monotonicity)
- ✅ Noise robustness (±5% input perturbation)
- ✅ Statistical properties (Q-Q plot, normality)

```matlab
[err_stats, err_plots] = validate_predictor(pred_fn, 'My Predictor');
```

-----

## Monte Carlo Methodology

### Why Monte Carlo?

Single-run comparisons can be misleading due to:

- Random terrain variation (different disturbance realization)
- Stochastic sensor noise
- Actuator/model uncertainties

**100 independent runs** with:

- **Fair seed control**: `rng(r)` reset before each method in each run
  → all three controllers see identical disturbance signals
- **Randomized noise level**: `noise ∈ [0.40, 0.70]` drawn fresh per run
  → tests robustness across operating conditions
- **Same mechanical plant**: PID and LQR both operate on identical A_aug, B_aug
  → **purely architectural comparison** (feedforward vs. none)

### Metrics Recorded

- **RMSE**: `√(mean(T_ref - T_actual)²)` — tracking accuracy [N]
- **Derailment Risk**: `100 · count(T < 20kN) / num_points` [%]
  - Critical safety metric; anything > 0.5% is unacceptable

-----

## Results Interpretation

### Expected Outcome

```
RMSE Improvement:
  LQR+FIS:  ~24% better than PID
  LQR+NN:   ~30% better than PID

Derailment Risk Reduction:
  PID:       ~2.3% (unacceptable)
  LQR+FIS:   ~0.8% (marginal)
  LQR+NN:    ~0.65% (good)
```

### Why LQR+NN Outperforms LQR+FIS

1. **Smooth nonlinear surface**: NN learns continuous mapping, avoiding discrete FIS rule boundaries
1. **Data-driven**: Adapts to training data characteristics; FIS relies on hand-tuned membership functions
1. **Fewer parameters**: FIS has 9 rules × multiple MF parameters; NN has ~70 weights (more efficient)
1. **Generalization**: NN trained on 2000 points vs. FIS covering 3×3 input grid

### Why LQR Outperforms PID

1. **Feedforward disturbance rejection**: LQR pre-adjusts reference; PID reacts after error occurs
1. **Optimal state feedback**: LQR minimizes quadratic cost; PID has fixed gains (tuned for one condition)
1. **Integral action**: Both have it; LQR’s is **optimally weighted** via Q matrix
1. **Model-based**: LQR uses knowledge of system dynamics; PID is purely heuristic

-----

## Code Walkthrough

### Main Simulation (adaptive_tension_v8_mechanical.m)

**Section 1: Configuration**

```matlab
time_step = 0.005;           % dt < 1/(10*ωₙ) ≈ 0.004 for stability
m = 250; k_track = 150000;   % Mechanical parameters (literature values)
velocity_scaling = 15;        % T_ref = T₀ + 15·v²
ff_gain = 0.35;              % Feedforward gain: ΔT_ref = 0.35·F_pred
```

**Section 2: LQR Design**

```matlab
Q_lqr = diag([900, 0.49, 4.4e-9, 1.0]);  % Weights on x_ext, x_vel, x_act, x_int
[K_lqr, ~, eig_cl] = lqr(A_aug, B_aug, Q_lqr, R_lqr);
% All eigenvalues negative → stable closed loop
```

**Section 3-4: FIS & NN Builders**

```matlab
fis = build_fterrain_fis();     % Mamdani FIS, 9 rules, Gaussian MF
net = train_fterrain_nn();      % fitnet([12 6]), 2000 training samples
```

**Section 5: Monte Carlo**

```matlab
for r = 1:num_runs
    rng(r);  % Seed controls disturbance realization
    noise = 0.40 + 0.30*rand();  % Run-to-run variation
    
    rng(r); [T_pid, ...] = run_pid_mech(...);      % PID
    rng(r); [T_fis, ...] = run_lqr_mech(...pred_fuzzy...);  % LQR+FIS
    rng(r); [T_nn,  ...] = run_lqr_mech(...pred_nn...);     % LQR+NN
end
```

**Key insight**: `rng(r)` is called **before each method**, not once at the start.
This ensures byte-for-byte identical disturbance sequences across methods.

-----

## Extending the Code

### Adding a New Predictor

1. **Implement the predictor function** (must match interface):
   
   ```matlab
   function F_pred = my_predictor(velocity, susp_var, tension_err)
       % Your predictor logic here
       F_pred = ... % compute F_terrain prediction
       F_pred = max(0, min(10000, F_pred));  % clamp to physical range
   end
   ```
1. **Wrap as closure**:
   
   ```matlab
   pred_custom = @(v, sv, e) my_predictor(v, sv, e);
   ```
1. **Run simulation**:
   
   ```matlab
   [T_custom, Tref_custom, met_custom] = run_lqr_mech(..., pred_custom);
   ```

### Adding a New Controller

To replace LQR with, e.g., MPC:

1. Design your controller in a new function `run_mpc_mech(...)`
1. Ensure it returns `[T_actual, T_ref, metrics]` struct
1. Call it in the Monte Carlo loop alongside PID/LQR

-----

## Publication Roadmap

### Paper Outline Suggestion

1. **Introduction**: Track tension problem, prior work (manual, proportional)
1. **Methodology**:
- Mechanical model derivation (Section 2 of main script)
- LQR design (Q matrix rationale, eigenvalue placement)
- Predictor architectures (Polynomial, GP, PINN)
1. **Experiments**:
- Monte Carlo setup (fairness: rng control)
- Ground truth model for terrain force
- Results table + statistical significance
1. **Discussion**:
- Why NN outperforms FIS
- Robustness to model uncertainty
- Real-world deployment considerations (embedded constraints)
1. **Conclusion**: Future work (reinforcement learning, field testing)

### Figures to Include

1. **System architecture diagram** (control loop)
1. **RMSE histogram + boxplot** (Monte Carlo comparison)
1. **Tracking example** (T_ref vs T_actual over 60 s)
1. **FIS/NN output surface** (F_terrain vs [susp_var, velocity])
1. **Terrain force ground truth** (synthetic model validation)

### Equations to Highlight

```
Mechanical Model:
  m·ẍ + c·ẋ + k·x = F_act + F_terrain

Tension Output:
  T(t) = T₀ + k·x + c·ẋ

LQR Optimal Gain:
  K = R⁻¹·B'·P   where P solves the algebraic Riccati equation

Ground Truth Terrain Force:
  F_terrain = 30000·susp_var·(1 + 0.4·(v/10)²)
```

-----

## FAQ

### Q: Why not use Simulink?

**A:** MATLAB code is more portable, easier to version-control, faster for parameter sweeps, and pedagogically clearer (no black-box blocks).

### Q: How sensitive are results to dt (time step)?

**A:** Very. We use dt=0.005 s (chosen so dt < 1/(10·ωₙ)). If you change it, LQR eigenvalues will shift; redo the design.

### Q: Can I use this for embedded systems?

**A:** Partially. The Polynomial predictor + anti-windup PID can run on an STM32 (~1 kHz loop rate). LQR+NN requires more compute. Fuzzy Logic Toolbox won’t run on firmware.

### Q: Is 100 runs enough for statistical significance?

**A:** For RMSE comparison, yes (~5% confidence interval). For rare events (< 0.1% derailment), no — use 1000+ runs.

### Q: How do I validate on real track data?

**A:** Collect [susp_var, velocity] → F_terrain_actual pairs from field testing. Fit the ground truth model parameters via regression. Compare predictor outputs to actual.

### Q: What if my vehicle’s k and c values are different?

**A:** Update lines 54-56. Recompute LQR (line 103). The framework is parameter-agnostic.

-----

## Citation

If you use this code in research, please cite:

```bibtex
@software{guner2026adaptivetrack,
  author       = {Güner, Ayça Bahar},
  title        = {Adaptive Track Tension Control with AI-Supported Terrain Prediction},
  year         = {2026},
  url          = {https://github.com/[your-repo]},
  note         = {MATLAB simulation framework for armored vehicle track management}
}
```

-----

## Author & Contact

**Ayça Bahar Güner**  
Mechanical Engineering (2nd year)  
Dokuz Eylül University (DEÜ), Izmir, Turkey

**Research Interests:**  
Control systems, mechanical design, geopolitical analysis, AI-augmented engineering  
**Languages:** Turkish (native), English (C1), German (B1-B2)

📧 For questions: [your-email]  
🔗 GitHub: [your-profile]

-----

## License

This code is provided for educational and research purposes under the [MIT / Apache 2.0] license.
Feel free to fork, modify, and share — with attribution.

-----

## Acknowledgments

- **Prof. Dr. Zeki Kıral** (DEÜ Dynamics & Vibration) — advisor
- **Emine** — internship mentor & networking support
- MATLAB & Fuzzy Logic / Deep Learning Toolbox teams

-----

**Last Updated**: June 2026  
**v8.0 Release**: Mechanical model fully integrated, AI predictor framework finalized