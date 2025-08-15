# DQN Self-Driving Car 🚗🤖

A 2D self-driving car simulator powered by **Deep Q-Learning (DQN)**.  
The car learns to navigate a custom race track using **raycasting sensors**, a **10-dimensional state vector**, and **reward shaping** to complete laps efficiently while avoiding collisions.

---

## 📌 Features
- **2D Track Environment** with static walls and checkpoints
- **Raycasting Sensors** for obstacle detection (5 beams)
- **Discrete Action Space**: accelerate, brake, turn left, turn right, coast
- **Deep Q-Network (DQN)** with:
  - Epsilon-greedy exploration (decay from 1.0 → 0.1)
  - Replay buffer
  - Target network updates
- **Reward Shaping** for progress, collisions, and lap completion
- **Performance Tracking** with training curves

---

## 🛠 Tech Stack
- **Language:** Python 3
- **Libraries:** PyTorch, Pygame, NumPy, Matplotlib
  

---

## 📂 Repository Structure
── docs/
│ ├── project_report.md # Detailed research-style report
│ ├── architecture_diagram.png # System flowchart
│ ├── state_space_explained.png # State vector diagram
│ ├── training_curve.png # Episode vs. total reward plot
│ ├── future_work.md # Planned improvements
├── self_fast.py
├── requirements.txt
└── README.md # This file


---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

pip install -r requirements.txt

python self_fast.py
<img width="900" height="731" alt="Screenshot 2025-08-16 022302" src="https://github.com/user-attachments/assets/1c718aa4-d443-48e7-9ed5-68955711acc6" />
<img width="998" height="784" alt="Screenshot 2025-08-16 022343" src="https://github.com/user-attachments/assets/dfb727e5-85c0-4fa6-8a6c-d1f01692632f" />
<img width="997" height="790" alt="Screenshot 2025-08-16 022332" src="https://github.com/user-attachments/assets/955e9483-bd5d-4618-bf5e-fecb404e1bae" />
<img width="993" height="786" alt="Screenshot 2025-08-16 022315" src="https://github.com/user-attachments/assets/578974ee-7597-4e03-839e-40858a9c2164" />


📚 Citation

If you use this project in your research, please cite:

@misc{dqn_self_driving_car,
  title={DQN Self-Driving Car: 2D Reinforcement Learning Simulator},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/dqn-self-driving-car}
}


