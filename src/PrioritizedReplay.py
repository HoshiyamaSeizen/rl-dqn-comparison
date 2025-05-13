import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # степень приоритезации (0 = равномерно)
        self.beta = beta  # компенсация смещения
        self.beta_increment = beta_increment
        self.max_priority = 1.0  # начальный приоритет для новых переходов

    def push(self, *transition):
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)  # новым переходам даём максимальный приоритет

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # нормализация

        state, action, reward, next_state, done = zip(*samples)
        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + 1e-5)  # обновляем приоритеты
        self.max_priority = max(self.priorities)  # обновляем максимальный приоритет

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers):
        super().__init__()
        net = []
        last_dim = input_dim
        for l in layers:
            net.append(nn.Linear(last_dim, l))
            net.append(nn.ReLU())
            last_dim = l
        net.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

class PrioritizedReplayAgent:
    def __init__(self, state_dim, action_dim, layer_cfg, gamma, epsilon, epsilon_decay):
        self.q_net = QNetwork(state_dim, action_dim, layer_cfg)
        self.target_net = QNetwork(state_dim, action_dim, layer_cfg)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.buffer = PrioritizedReplayBuffer() # <-- Заменяем на PrioritizedReplayBuffer
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
        s, a, r, s2, d, indices, weights = self.buffer.sample(128)  # <-- Добавляем weights
        s, a, r, s2, d, weights = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device), weights.to(self.device)

        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * self.target_net(s2).max(1)[0] * (1 - d)
        
        td_errors = (target - q_vals).abs().detach().cpu().numpy()  # для обновления приоритетов
        loss = (weights * (q_vals - target) ** 2).mean()  # взвешенная MSE

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors)  # обновляем приоритеты
        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.05)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())