# Reinforcement Learning for 2D Self-Driving Car Simulation

## Abstract
We present a 2D simulation of a self-driving car trained using Deep Q-Learning (DQN) to navigate a custom race track. The car perceives its environment through raycasting sensors and optimizes its policy to complete laps efficiently while avoiding collisions.

## Introduction & Motivation
Autonomous driving is a challenging problem involving perception, decision-making, and control. While 3D simulators like Carla exist, lightweight 2D environments allow rapid prototyping of RL algorithms without heavy compute.

## Related Work
- **OpenAI Gym** – standardized RL interface.
- **Carla Simulator** – photorealistic autonomous driving simulation.
- **Deep Q-Networks (Mnih et al., 2015)** – introduced stable Q-learning with replay buffers.

## Methodology
### Environment
- **Track Representation:** Static walls defined as line segments.
- **Sensors:** 5 raycasting distance measurements.
- **State Vector:** 10 features → `[sensor1, sensor2, sensor3, sensor4, sensor5, speed, angle, lap_time, checkpoint_index, extra_feature]`.
- **Action Space:** 5 discrete actions (accelerate, brake, turn left, turn right, coast).
- **Rewards:**
  - Progress toward next checkpoint: +1
  - Collision: -15
  - Off-track: -5
  - Lap completion: +50

### Algorithm
- **Deep Q-Network (DQN)** with:
  - Epsilon-greedy exploration (ε decays from 1.0 to 0.1)
  - Replay buffer size: 50,000
  - Target network update every 1000 steps
  - Learning rate: 1e-3
- Loss: MSE between predicted Q-values and TD target.

## Experiments
- **Episodes:** 1000
- **Hardware:** RTX 3060 GPU
- **Frameworks:** PyTorch, Pygame
- **Track Layout:** Simple oval for early learning.

## Results
- Learning curve shows improvement after ~300 episodes.
- Best performance: agent completes lap with 85% fewer collisions than random policy.

## Discussion
- DQN learns basic navigation but struggles with sharp turns.
- Reward shaping critical for stable learning.
- Lap time feature helped optimize speed control.

## Conclusion
This project demonstrates that a simple DQN can learn to navigate a 2D track effectively using low-dimensional sensor data.

---
