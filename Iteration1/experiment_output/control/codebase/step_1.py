# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import pandas as pd
import os

data_dir = "data/"

def lyapunov_phi(state):
    cos_theta = state[..., 0]
    theta_dot = state[..., 2]
    return (1.0 - cos_theta) + 0.5 * theta_dot ** 2

def lyapunov_phi_np(state):
    cos_theta = state[..., 0]
    theta_dot = state[..., 2]
    return (1.0 - cos_theta) + 0.5 * theta_dot ** 2

class LyapunovWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._prev_phi = None
        self._upright_steps = 0
        self._total_steps = 0
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = lyapunov_phi_np(obs)
        self._upright_steps = 0
        self._total_steps = 0
        return obs, info
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        new_phi = lyapunov_phi_np(obs)
        reward = self._prev_phi - new_phi
        self._prev_phi = new_phi
        theta = np.arctan2(obs[1], obs[0])
        is_upright = float(abs(theta) < 0.1)
        self._upright_steps += is_upright
        self._total_steps += 1
        info['upright'] = is_upright
        return obs, reward, terminated, truncated, info

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_scale=2.0):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -5, 2)
        return mean, log_std
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

class CriticA(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

class CriticB(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.f1_body = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.f1_head = nn.Linear(hidden_dim, 1)
        self.f2_body = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.f2_head = nn.Linear(hidden_dim, 1)
        for h in [self.f1_head, self.f2_head]:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        phi = lyapunov_phi(state).unsqueeze(-1)
        return phi + self.f1_head(self.f1_body(sa)), phi + self.f2_head(self.f2_body(sa))

if __name__ == '__main__':
    print("Training initialized.")