import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers):
        super().__init__()
        # Общая часть сети
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, layers[0]),
            nn.ReLU(),
        )
        # Ветка для V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], 1),
        )
        # Ветка для A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], output_dim),
        )

    def forward(self, x):
        features = self.feature_net(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
        return value + (advantages - advantages.mean(dim=-1, keepdim=True))

class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, layer_cfg, gamma, epsilon, epsilon_decay):
        self.q_net = DuelingQNetwork(state_dim, action_dim, layer_cfg)  # <-- Заменяем на DuelingQNetwork
        self.target_net = DuelingQNetwork(state_dim, action_dim, layer_cfg)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        self.target_net.to(self.device)
        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.q_net(state_tensor).argmax().item()

    def train_step(self):
        if len(self.buffer) < 128:
            return 0
        s, a, r, s2, d = self.buffer.sample(128)
        s, a, r, s2, d = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device)

        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * self.target_net(s2).max(1)[0] * (1 - d)
        loss = nn.MSELoss()(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.05)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())