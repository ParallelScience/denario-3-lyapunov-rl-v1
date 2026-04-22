# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import random

class LyapunovRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        theta_t = np.arctan2(self.prev_obs[1], self.prev_obs[0])
        phi_t = (1.0 - np.cos(theta_t)) + 0.5 * (self.prev_obs[2] ** 2)
        theta_tp1 = np.arctan2(obs[1], obs[0])
        phi_tp1 = (1.0 - np.cos(theta_tp1)) + 0.5 * (obs[2] ** 2)
        lyapunov_reward = phi_t - phi_tp1
        self.prev_obs = obs
        info['original_reward'] = reward
        info['phi'] = phi_tp1
        return obs, lyapunov_reward, terminated, truncated, info

def compute_phi_tensor(state):
    theta = torch.atan2(state[:, 1], state[:, 0])
    phi = (1.0 - torch.cos(theta)) + 0.5 * (state[:, 2] ** 2)
    return phi.unsqueeze(1)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, structured=False):
        super(QNetwork, self).__init__()
        self.structured = structured
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)
        if self.structured:
            torch.nn.init.uniform_(self.linear3.weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.linear3.bias, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.linear6.weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.linear6.bias, -1e-3, 1e-3)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        q1 = self.linear3(x1)
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        q2 = self.linear6(x2)
        if self.structured:
            phi = compute_phi_tensor(state)
            q1 = q1 + phi
            q2 = q2 + phi
        return q1, q2

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class ReplayMemory:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], 1.0 - self.dones[idxs])

    def __len__(self):
        return self.size

class SACArgs:
    def __init__(self):
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.target_update_interval = 1
        self.hidden_size = 256
        self.lr = 3e-4
        self.automatic_entropy_tuning = True
        self.structured = False

class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.target_update_interval = args.target_update_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, structured=args.structured).to(device=self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, structured=args.structured).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(args.alpha).to(self.device)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha.detach() * log_pi) - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = self.alpha.clone()
        if updates % self.target_update_interval == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    env = LyapunovRewardWrapper(env)
    obs, info = env.reset(seed=42)
    for i in range(5):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        phi = info['phi']
        assert phi >= 0, "Phi must be non-negative"
        obs = next_obs
    args = SACArgs()
    args.structured = False
    agent_A = SAC(env.observation_space.shape[0], env.action_space, args)
    memory_A = ReplayMemory(10000, env.observation_space.shape[0], env.action_space.shape[0])
    obs, _ = env.reset()
    for i in range(1000):
        action = agent_A.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        memory_A.push(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    agent_A.update_parameters(memory_A, 64, 1)
    args.structured = True
    agent_B = SAC(env.observation_space.shape[0], env.action_space, args)
    memory_B = ReplayMemory(10000, env.observation_space.shape[0], env.action_space.shape[0])
    obs, _ = env.reset()
    for i in range(1000):
        action = agent_B.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        memory_B.push(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    agent_B.update_parameters(memory_B, 64, 1)
    with open("data/step_1_success.txt", "w") as f:
        f.write("Step 1 completed successfully.")