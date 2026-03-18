# Adaptive Continuous Robot Control in Varying Resistance Environments Using Reinforcement Learning

This project studies reinforcement-learning-based control of a continuous multi-link robotic arm operating in environments with spatially varying resistance, with a particular focus on air–water interface scenarios.

The goal is to train a control policy that enables the robot arm to reach target positions accurately while adapting to changing drag and hydrodynamic effects, and at the same time reducing total energy consumption and motion time.

## Overview

Robotic motion across different media, such as air and water, introduces abrupt changes in resistance, drag, and added-mass effects. These discontinuities make conventional controllers difficult to apply reliably. This project builds a full reinforcement learning framework in MuJoCo to address this problem.

The system models a 10-DOF planar robotic arm moving in a two-fluid environment. During motion, different arm segments may lie in different media, resulting in time-varying hydrodynamic loads along the robot body. The controller must adapt online to these varying resistive forces while completing a target-reaching task efficiently. 

## Project Goals

- Build a MuJoCo-based simulation environment for a multi-link robot arm in a two-medium setting
- Model varying fluid resistance across an air–water interface
- Train a reinforcement learning agent for adaptive target reaching
- Compare RL performance with inverse kinematics (IK) baselines
- Evaluate control quality in terms of accuracy, stability, and energy consumption

## Problem Setting

### Robot

- 10-DOF planar serial manipulator
- Operates in the x-z plane
- End-effector target reaching task in Cartesian space

### Environment

The workspace is divided into two horizontal fluid regions:

- **Fluid 1 (water)**: below the interface
- **Fluid 2 (air)**: above the interface

A smooth transition function is used near the interface to avoid discontinuities in the fluid parameters. At every simulation step, each arm segment is assigned effective hydrodynamic parameters based on its position relative to the interface.

### Task

At the beginning of each episode, a target position is sampled from a reachable workspace. The controller must move the end-effector to the target within a fixed horizon while minimizing:

- final reaching error
- total energy consumption
- excessive collision and unstable motion

## Method

## Reinforcement Learning Framework

The RL framework contains the following components:

### Observation Space

The observation vector includes:

- joint positions
- joint velocities
- end-effector Cartesian position
- end-effector position error relative to the target
- remaining episode time

This combines both robot-state information and task-state information for policy learning.

### Action Space

The final controller uses:

- **10-dimensional desired joint velocities**

The commanded velocities are integrated inside the environment and tracked by a built-in PID controller, which produces the final joint torques applied in MuJoCo.

Compared with direct torque control, velocity-based action design improves training stability and results in smoother motions.

### Reward Function

The reward includes several terms:

- task progress reward
- success bonus
- energy penalty
- collision penalty
- time reward

This reward design encourages the agent to reach the target accurately while avoiding wasteful motion and reducing total energy consumption.

## Results

The experiments include the following settings:

### 1. Single-medium experiments

The learned policy was first evaluated in a single-fluid environment to study convergence stability and energy efficiency under constant resistance.

#### Fixed target reaching
The velocity-based controller achieved stable and smooth reaching behavior.

#### Random target reaching
Velocity-based control showed more stable and energy-efficient motion than torque-based control.

### 2. Two-medium experiments

The learned policy was then tested in the main air–water interface scenario.

- The velocity-based RL agent maintained stable behavior across the interface.
- It successfully reached the target despite abrupt changes in drag and hydrodynamic load.

### 3. Comparison with IK baseline

The RL controller was compared with a traditional inverse kinematics controller.

<p align="center">
  <img src="images/compare ik.png" alt="RL versus IK comparison" width="65%">
</p>

Main findings:

- IK showed oscillation and poor adaptation to drag differences.
- RL generated smoother and more drag-aware trajectories.
- RL consumed significantly less energy than the IK baseline.

According to the report, the RL policy achieved much lower energy consumption than IK in the two-medium scenario.

## Key Findings

- Action-space design strongly affects learning stability.
- Velocity control is significantly more stable than direct torque control in two-fluid dynamics.
- RL can learn environment-aware strategies without explicit fluid labels.
- Energy-aware reward shaping improves motion efficiency.
- Compared with IK, RL adapts better to varying resistance environments.

## Dependencies

This project is built on the following libraries:

- Python 3.x
- MuJoCo
- NumPy
- SciPy
- PyTorch
- Stable-Baselines3
- Gymnasium
- Matplotlib
