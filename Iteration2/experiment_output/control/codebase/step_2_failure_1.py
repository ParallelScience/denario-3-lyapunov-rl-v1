# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from step_1 import (lyapunov_phi_np, LyapunovPendulumWrapper, ActorNetwork, CriticNetworkA, CriticNetworkB)
DATA_DIR = 'data/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4
ROLLOUT_LEN = 2048
MINIBATCH_SIZE = 64
N_EPOCHS = 4
TOTAL_STEPS = 100000
N_SEEDS = 5
ACTION_SCALE = 2.0
def make_env(seed):
    env = gym.make('Pendulum-v1')
    env = LyapunovPendulumWrapper(env)
    env.reset(seed=seed)
    return env
def compute_gae(rewards, values, dones, next_value, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns
def ppo_update(actor, critic, actor_opt, critic_opt, states_t, actions_t, log_probs_old_t, advantages_t, returns_t, condition, n_epochs, minibatch_size):
    T = states_t.shape[0]
    actor.train()
    critic.train()
    for _ in range(n_epochs):
        indices = torch.randperm(T, device=DEVICE)
        for start in range(0, T, minibatch_size):
            idx = indices[start:start + minibatch_size]
            s_mb, a_mb, lp_old_mb, adv_mb, ret_mb = states_t[idx], actions_t[idx], log_probs_old_t[idx], advantages_t[idx], returns_t[idx]
            dist = actor.get_dist(s_mb)
            log_probs_new = dist.log_prob(a_mb).sum(-1)
            ratio = torch.exp(log_probs_new - lp_old_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_mb
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * dist.entropy().sum(-1).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_opt.step()
            v_pred = critic(s_mb) if condition == 'A' else critic.get_value(s_mb)
            critic_loss = VALUE_COEF * nn.functional.mse_loss(v_pred, ret_mb)
            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()
def run_training():
    for condition in ['A', 'B']:
        for seed in range(N_SEEDS):
            env = make_env(seed)
            actor = ActorNetwork(3, 1).to(DEVICE)
            critic = (CriticNetworkA(3) if condition == 'A' else CriticNetworkB(3)).to(DEVICE)
            actor_opt = optim.Adam(actor.parameters(), lr=LR)
            critic_opt = optim.Adam(critic.parameters(), lr=LR)
            steps_done = 0
            while steps_done < TOTAL_STEPS:
                states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
                obs, _ = env.reset()
                for _ in range(ROLLOUT_LEN):
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    dist = actor.get_dist(obs_t)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1).item()
                    val = (critic(obs_t) if condition == 'A' else critic.get_value(obs_t)).item()
                    action_np = torch.clamp(action, -ACTION_SCALE, ACTION_SCALE).cpu().numpy().flatten()
                    next_obs, reward, term, trunc, _ = env.step(action_np)
                    states.append(obs); actions.append(action_np); log_probs.append(log_prob); rewards.append(reward); values.append(val); dones.append(float(term or trunc))
                    obs = next_obs
                next_val = (critic(torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)) if condition == 'A' else critic.get_value(torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))).item()
                adv, ret = compute_gae(np.array(rewards), np.array(values), np.array(dones), next_val, GAMMA, LAM)
                ppo_update(actor, critic, actor_opt, critic_opt, torch.tensor(np.array(states), device=DEVICE), torch.tensor(np.array(actions), device=DEVICE), torch.tensor(np.array(log_probs), device=DEVICE), torch.tensor(adv, device=DEVICE), torch.tensor(ret, device=DEVICE), condition, N_EPOCHS, MINIBATCH_SIZE)
                steps_done += ROLLOUT_LEN
            torch.save(actor.state_dict(), os.path.join(DATA_DIR, 'actor_' + condition + '_' + str(seed) + '.pth'))
            torch.save(critic.state_dict(), os.path.join(DATA_DIR, 'critic_' + condition + '_' + str(seed) + '.pth'))
if __name__ == '__main__':
    run_training()