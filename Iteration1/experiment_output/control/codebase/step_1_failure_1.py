# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

def lyapunov_phi(state):
    if isinstance(state, np.ndarray):
        theta = np.arctan2(state[..., 1], state[..., 0])
        theta_dot = state[..., 2]
        return (1.0 - np.cos(theta)) + 0.5 * (theta_dot ** 2)
    elif isinstance(state, torch.Tensor):
        theta = torch.atan2(state[..., 1], state[..., 0])
        theta_dot = state[..., 2]
        return (1.0 - torch.cos(theta)) + 0.5 * (theta_dot ** 2)
    else:
        raise TypeError("State must be a numpy array or torch tensor.")

class LyapunovRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_phi = None
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_phi = lyapunov_phi(obs)
        return obs, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_phi = lyapunov_phi(obs)
        lyapunov_reward = self.prev_phi - current_phi
        self.prev_phi = current_phi
        return obs, lyapunov_reward, terminated, truncated, info

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std
    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mu) * self.max_action

class CriticA(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticA, self).__init__()
        self.q1_net = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2_net = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2

class CriticB(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticB, self).__init__()
        self.q1_net = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2_net = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        nn.init.zeros_(self.q1_net[-1].weight)
        nn.init.zeros_(self.q1_net[-1].bias)
        nn.init.zeros_(self.q2_net[-1].weight)
        nn.init.zeros_(self.q2_net[-1].bias)
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        f1 = self.q1_net(sa)
        f2 = self.q2_net(sa)
        phi = lyapunov_phi(state).unsqueeze(1)
        q1 = phi + f1
        q2 = phi + f2
        return q1, q2

class SAC:
    def __init__(self, state_dim, action_dim, max_action, critic_class, device, lr=3e-4, gamma=0.99, tau=0.005):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = critic_class(state_dim, action_dim).to(self.device)
        self.critic_target = critic_class(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()
    def train(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        pi, log_prob, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.log_alpha.exp().detach() * log_prob) - min_q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

def test_sac_1_episode():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('Pendulum-v1')
    env = LyapunovRewardWrapper(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent_a = SAC(state_dim, action_dim, max_action, CriticA, device)
    replay_buffer_a = ReplayBuffer(10000, state_dim, action_dim, device)
    state, _ = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    steps = 0
    while not (done or truncated):
        action = agent_a.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer_a.add(state, action, reward, next_state, float(done))
        state = next_state
        episode_reward += reward
        steps += 1
        if replay_buffer_a.size > 64:
            agent_a.train(replay_buffer_a, batch_size=64)
    print("Test CriticA 1 episode completed. Reward: " + str(round(episode_reward, 2)) + ", Steps: " + str(steps))
    agent_b = SAC(state_dim, action_dim, max_action, CriticB, device)
    replay_buffer_b = ReplayBuffer(10000, state_dim, action_dim, device)
    state, _ = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    steps = 0
    while not (done or truncated):
        action = agent_b.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer_b.add(state, action, reward, next_state, float(done))
        state = next_state
        episode_reward += reward
        steps += 1
        if replay_buffer_b.size > 64:
            agent_b.train(replay_buffer_b, batch_size=64)
    print("Test CriticB 1 episode completed. Reward: " + str(round(episode_reward, 2)) + ", Steps: " + str(steps))

if __name__ == '__main__':
    test_sac_1_episode()