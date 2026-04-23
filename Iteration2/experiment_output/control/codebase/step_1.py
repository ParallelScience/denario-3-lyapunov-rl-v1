# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import os

def lyapunov_phi(states_tensor):
    cos_theta = states_tensor[:, 0]
    thetadot = states_tensor[:, 2]
    phi = (1.0 - cos_theta) + 0.5 * thetadot ** 2
    return phi

def lyapunov_phi_np(state):
    cos_theta = state[0]
    thetadot = state[2]
    return float((1.0 - cos_theta) + 0.5 * thetadot ** 2)

class LyapunovPendulumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._prev_phi = None
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = lyapunov_phi_np(obs)
        return obs, info
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        new_phi = lyapunov_phi_np(obs)
        reward = self._prev_phi - new_phi
        self._prev_phi = new_phi
        return obs, reward, terminated, truncated, info

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.mean_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    def forward(self, x):
        features = self.net(x)
        mean = self.mean_head(features)
        return mean, self.log_std
    def get_dist(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)

class CriticNetworkA(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

class CriticNetworkB(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)
    def get_value(self, x):
        return lyapunov_phi(x) + self.forward(x)

if __name__ == '__main__':
    print('Training script initialized.')