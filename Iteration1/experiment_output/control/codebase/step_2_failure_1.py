# filename: codebase/step_2.py
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
from step_1 import (lyapunov_phi, lyapunov_phi_np, LyapunovWrapper, ReplayBuffer, Actor, CriticA, CriticB)

data_dir = 'data/'
TOTAL_STEPS = 100000
REPLAY_BUFFER_SIZE = 100000
BATCH_SIZE = 256
LR = 0.0003
GAMMA = 0.99
TAU = 0.005
HIDDEN_DIM = 256
WARMUP_STEPS = 1000
SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = ['A', 'B']
EVAL_EPISODES = 20
GRID_SIZE = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

def train_sac(condition, seed, total_steps=TOTAL_STEPS, device=DEVICE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = LyapunovWrapper(gym.make('Pendulum-v1'))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    actor = Actor(state_dim, action_dim, HIDDEN_DIM, action_scale).to(device)
    if condition == 'A':
        critic = CriticA(state_dim, action_dim, HIDDEN_DIM).to(device)
        critic_target = CriticA(state_dim, action_dim, HIDDEN_DIM).to(device)
    else:
        critic = CriticB(state_dim, action_dim, HIDDEN_DIM).to(device)
        critic_target = CriticB(state_dim, action_dim, HIDDEN_DIM).to(device)
    critic_target.load_state_dict(critic.state_dict())
    actor_opt = optim.Adam(actor.parameters(), lr=LR)
    critic_opt = optim.Adam(critic.parameters(), lr=LR)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=LR)
    target_entropy = -float(action_dim)
    replay = ReplayBuffer(REPLAY_BUFFER_SIZE, state_dim, action_dim, device)
    episode_log = []
    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    ep_steps = 0
    ep_upright = 0
    ep_critic_losses = []
    for step in range(total_steps):
        if step < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _ = actor.sample(s_t)
                action = action.cpu().numpy()[0]
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay.add(obs, action, reward, next_obs, float(done and not truncated))
        ep_return += reward
        ep_steps += 1
        ep_upright += info.get('upright', 0.0)
        if step >= WARMUP_STEPS and replay.size >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)
            with torch.no_grad():
                next_actions, next_log_probs = actor.sample(next_states)
                q1_next, q2_next = critic_target(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next) - log_alpha.exp() * next_log_probs
                q_target = rewards + GAMMA * (1.0 - dones) * q_next
            q1, q2 = critic(states, actions)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            ep_critic_losses.append(critic_loss.item())
            curr_actions, curr_log_probs = actor.sample(states)
            q1_pi, q2_pi = critic(states, curr_actions)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (log_alpha.exp().detach() * curr_log_probs - q_pi).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            alpha_loss = -(log_alpha.exp() * (curr_log_probs + target_entropy).detach()).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()
            soft_update(critic_target, critic, TAU)
        obs = next_obs
        if done:
            upright_frac = ep_upright / ep_steps if ep_steps > 0 else 0.0
            episode_log.append({'step': step + 1, 'episode_return': ep_return, 'upright_fraction': upright_frac})
            obs, _ = env.reset()
            ep_return = 0.0
            ep_steps = 0
            ep_upright = 0
    env.close()
    return actor, critic, episode_log

if __name__ == '__main__':
    results = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            actor, critic, log = train_sac(cond, seed)
            df = pd.DataFrame(log)
            df.to_csv(os.path.join(data_dir, 'log_' + cond + '_' + str(seed) + '.csv'), index=False)
            print('Finished training condition ' + cond + ' seed ' + str(seed))