# adaptive-tension-track-system
adaptive-tension-track-systems
# AI-Supported Adaptive Tension Track System (Proof of Concept)

This repository contains the 1D Proof of Concept (PoC) simulation for an **AI-Supported Adaptive Track Tension System** for armored vehicles. This work is currently a research project proposal being developed by Ayça Bahar Güner as an undergraduate student at **Dokuz Eylül University, Department of Mechanical Engineering**.

## Project Overview
The mobility of tracked armored vehicles is highly dependent on the mechanical interaction between the track and the terrain. Maintaining optimal track tension is critical for structural integrity and transmission efficiency. This project proposes an autonomous actuator system that processes real-time vehicle kinematics and terrain profile data to dynamically optimize track tension.

## Repository Contents
- `adaptive_tension_control.m`: The core MATLAB script. This simulation uses a PID controller with **Anti-Windup (Clamping)** logic to prevent actuator saturation issues while adjusting track tension across varying terrain conditions (Asphalt, Mud, Rocky).

## Key Features
- **Dynamic Terrain Modeling:** Simulates transitions between different terrain types with associated friction coefficients.
- **Robust PID Control:** Implements anti-windup protection to maintain system stability when the actuator reaches physical limits.
- **Control Signal Visualization:** Tracks actuator input signals against physical valve pressure limits to demonstrate controller stability.
- **Data-Driven Approach:** Designed as a foundational stage for future integration with machine learning models and digital twin simulations.

## How to Run
1. Ensure you have **MATLAB** installed.
2. Clone this repository or download `adaptive_tension_control.m`.
3. Run the script in MATLAB.
4. The simulation will generate 4 comprehensive plots illustrating:
    - Vehicle Kinematics (Sensor Data)
    - Terrain Friction Coefficient
    - PID Performance (Target vs. Actual Tension)
    - Actuator Control Signal with Physical Saturation Limits

## Future Research Goals
This PoC is currently being expanded into a full-scale research project involving:
- **Digital Twin Development:** Implementing Multi-Body Dynamics in Simulink.
- **Autonomous Optimization:** Replacing heuristic coefficients ($\alpha, \beta$) with optimized weights using Genetic Algorithms (GA) or Particle Swarm Optimization (PSO).
- **Structural Validation:** Integrating FEA (ANSYS) for fatigue and stress analysis.

## License
This project is for academic research and educational purposes.

---
*Developed by Ayça Bahar Güner | Dokuz Eylül University, Faculty of Engineering | May 2026*
