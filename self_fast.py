import pygame
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------- Pygame Setup -----------------
pygame.init()
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Self-Driving Car RL")
clock = pygame.time.Clock()

# ----------------- Track & Walls -----------------
walls = [
    ((50, 50), (750, 50)),
    ((750, 50), (750, 550)),
    ((750, 550), (50, 550)),
    ((50, 550), (50, 50)),
    ((150, 150), (650, 150)),
    ((650, 150), (650, 450)),
    ((650, 450), (150, 450)),
    ((150, 450), (150, 150))
]

checkpoints = [(400,100), (700,300), (400,500), (100,300)]

# ----------------- Raycasting -----------------
def line_intersection(p0, p1, p2, p3):
    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]
    denom = s10_x * s32_y - s32_x * s10_y
    if denom == 0: return None
    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]
    t = (s32_x * s02_y - s32_y * s02_x) / denom
    u = (s10_x * s02_y - s10_y * s02_x) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p0[0] + t * s10_x, p0[1] + t * s10_y)
    return None

def get_sensor_data(car, walls, ray_count=5, ray_length=100):
    distances = []
    for i in range(-ray_count//2, ray_count//2 + 1):
        angle = car.angle + i * (180 / ray_count)
        end = (car.x + ray_length*np.cos(np.radians(angle)),
               car.y + ray_length*np.sin(np.radians(angle)))
        min_dist = ray_length
        for wall in walls:
            inter = line_intersection((car.x, car.y), end, wall[0], wall[1])
            if inter:
                dist = np.sqrt((inter[0]-car.x)**2 + (inter[1]-car.y)**2)
                if dist < min_dist:
                    min_dist = dist
        distances.append(min_dist)
    return np.array(distances, dtype=np.float32)

# ----------------- Car Class -----------------
class Car:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.angle = 0.0
        self.speed = 0.0
        self.ray_count = 5
        self.ray_length = 100
        self.checkpoint_idx = 0
        self.lap_time = 0.0

    def move(self, action):
        if action == 0: self.angle -= 5
        if action == 1: self.angle += 5
        if action == 2: self.speed += 0.5
        if action == 3: self.speed -= 0.5
        self.x += self.speed * np.cos(np.radians(self.angle))
        self.y += self.speed * np.sin(np.radians(self.angle))
        self.speed *= 0.95
        self.lap_time += 1/60

    def draw(self, win):
        pygame.draw.rect(win, (255,0,0), (self.x, self.y, 20, 10))
        for i in range(-self.ray_count//2, self.ray_count//2 + 1):
            angle = self.angle + i * (180 / self.ray_count)
            end = (self.x + self.ray_length*np.cos(np.radians(angle)),
                   self.y + self.ray_length*np.sin(np.radians(angle)))
            pygame.draw.line(win, (0,255,0), (self.x, self.y), end, 1)

# ----------------- DQN -----------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

# ----------------- Hyperparameters -----------------
dummy_car = Car(0,0)
STATE_DIM = len(np.concatenate([
    get_sensor_data(dummy_car, walls),
    np.array([dummy_car.speed, dummy_car.angle, dummy_car.lap_time, dummy_car.checkpoint_idx], dtype=np.float32),
    np.array([0.0], dtype=np.float32)  # extra feature for 10
]))
ACTION_DIM = 5
BUFFER_CAPACITY = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
epsilon = EPSILON_START

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(STATE_DIM, ACTION_DIM).to(device)
target_net = DQN(STATE_DIM, ACTION_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

# ----------------- Reward -----------------
def compute_reward(car):
    reward = 0.0
    done = False
    for wall in walls:
        if line_intersection((car.x, car.y), (car.x+5, car.y+5), wall[0], wall[1]):
            reward -= 10
            done = True
    checkpoint = checkpoints[car.checkpoint_idx]
    if np.linalg.norm(np.array([car.x, car.y])-np.array(checkpoint)) < 30:
        reward += 5
        car.checkpoint_idx = (car.checkpoint_idx + 1) % len(checkpoints)
    reward += car.speed * 0.1
    return reward, done

# ----------------- Training Loop -----------------
num_episodes = 1000

for episode in range(num_episodes):
    car = Car(100,100)
    state = np.concatenate([
        get_sensor_data(car, walls)/100.0,       # normalize distances
        np.array([car.speed/10, car.angle/360, car.lap_time/60, car.checkpoint_idx/len(checkpoints)], dtype=np.float32),
        np.array([0.0], dtype=np.float32)        # extra normalized feature
    ])
    total_reward = 0.0
    done = False

    while not done:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, ACTION_DIM-1)
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()

        car.move(action)
        next_state = np.concatenate([
            get_sensor_data(car, walls)/100.0,
            np.array([car.speed/10, car.angle/360, car.lap_time/60, car.checkpoint_idx/len(checkpoints)], dtype=np.float32),
            np.array([0.0], dtype=np.float32)
        ])
        reward, done = compute_reward(car)
        total_reward += reward
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        # ----------------- Training -----------------
        if len(replay_buffer) > BATCH_SIZE:
            s, a, r, s_, d = replay_buffer.sample(BATCH_SIZE)
            s = torch.from_numpy(s).float().to(device)
            s_ = torch.from_numpy(s_).float().to(device)
            a = torch.from_numpy(a).long().unsqueeze(1).to(device)
            r = torch.from_numpy(r).float().unsqueeze(1).to(device)
            d = torch.from_numpy(d).float().unsqueeze(1).to(device)

            q_values = policy_net(s).gather(1,a)
            with torch.no_grad():
                # Double DQN target
                next_actions = policy_net(s_).argmax(dim=1, keepdim=True)
                target_q = r + GAMMA * target_net(s_).gather(1,next_actions) * (1-d)
            loss = nn.MSELoss()(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # ----------------- Draw -----------------
        win.fill((0,0,0))
        for wall in walls:
            pygame.draw.line(win, (255,255,255), wall[0], wall[1], 3)
        for cp in checkpoints:
            pygame.draw.circle(win, (0,0,255), cp, 10)
        car.draw(win)
        pygame.display.update()

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

pygame.quit()
