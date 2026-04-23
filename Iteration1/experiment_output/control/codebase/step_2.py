# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['OMP_NUM_THREADS'] = '2'
sys.path.insert(0, os.path.abspath('codebase'))
import numpy as np
import torch
import gymnasium as gym
import pandas as pd
from step_1 import (lyapunov_phi_np, LyapunovWrapper, Actor, CriticA, CriticB)

data_dir = 'data/'
SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = ['A', 'B']
EVAL_EPISODES = 20
GRID_SIZE = 100
HIDDEN_DIM = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_actor(condition, seed, device):
    actor = Actor(state_dim=3, action_dim=1, hidden_dim=HIDDEN_DIM, action_scale=2.0).to(device)
    sd = torch.load(os.path.join(data_dir, 'actor_' + condition + '_' + str(seed) + '.pt'), map_location=device)
    actor.load_state_dict(sd)
    actor.eval()
    return actor

def load_critic(condition, seed, device):
    if condition == 'A':
        critic = CriticA(state_dim=3, action_dim=1, hidden_dim=HIDDEN_DIM).to(device)
    else:
        critic = CriticB(state_dim=3, action_dim=1, hidden_dim=HIDDEN_DIM).to(device)
    sd = torch.load(os.path.join(data_dir, 'critic_' + condition + '_' + str(seed) + '.pt'), map_location=device)
    critic.load_state_dict(sd)
    critic.eval()
    return critic

def evaluate_policy(actor, seed, n_episodes=EVAL_EPISODES, device=DEVICE):
    env = LyapunovWrapper(gym.make('Pendulum-v1'))
    returns = []
    uprights = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        ep_ret = 0.0
        ep_up = 0.0
        ep_steps = 0
        done = False
        while not done:
            with torch.no_grad():
                s_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _ = actor.sample(s_t)
                action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            ep_up += info.get('upright', 0.0)
            ep_steps += 1
            done = terminated or truncated
        returns.append(ep_ret)
        uprights.append(ep_up / ep_steps if ep_steps > 0 else 0.0)
    env.close()
    return returns, uprights

def compute_sample_efficiency(episode_log, threshold=0.9):
    steps = episode_log['step'].values
    returns = episode_log['episode_return'].values
    window = 10
    if len(returns) < window:
        return None
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
    smoothed_steps = steps[window - 1:]
    max_val = np.max(smoothed)
    if max_val <= 0:
        return None
    target = threshold * max_val
    indices = np.where(smoothed >= target)[0]
    if len(indices) == 0:
        return None
    return int(smoothed_steps[indices[0]])

def build_state_grid(grid_size=GRID_SIZE):
    theta_vals = np.linspace(-np.pi, np.pi, grid_size)
    thetadot_vals = np.linspace(-8.0, 8.0, grid_size)
    TH, TD = np.meshgrid(theta_vals, thetadot_vals, indexing='ij')
    cos_th = np.cos(TH).astype(np.float32)
    sin_th = np.sin(TH).astype(np.float32)
    td_flat = TD.ravel().astype(np.float32)
    states_np = np.stack([cos_th.ravel(), sin_th.ravel(), td_flat], axis=-1)
    phi_grid = (1.0 - np.cos(TH)) + 0.5 * TD ** 2
    return states_np, phi_grid, theta_vals, thetadot_vals, TH, TD

def eval_q_grid(critic, states_np, grid_size=GRID_SIZE, device=DEVICE, batch=5000):
    states_t = torch.FloatTensor(states_np).to(device)
    n = states_t.shape[0]
    q_list = []
    with torch.no_grad():
        for i in range(0, n, batch):
            s_b = states_t[i:i + batch]
            a_b = torch.zeros(s_b.shape[0], 1, device=device)
            q1, q2 = critic(s_b, a_b)
            q_list.append(((q1 + q2) / 2.0).cpu().numpy())
    q_vals = np.concatenate(q_list, axis=0).reshape(grid_size, grid_size)
    return q_vals

def get_residual_f_grid(critic_b, states_np, grid_size=GRID_SIZE, device=DEVICE, batch=5000):
    states_t = torch.FloatTensor(states_np).to(device)
    n = states_t.shape[0]
    f_list = []
    with torch.no_grad():
        for i in range(0, n, batch):
            s_b = states_t[i:i + batch]
            a_b = torch.zeros(s_b.shape[0], 1, device=device)
            sa = torch.cat([s_b, a_b], dim=-1)
            f1 = critic_b.f1_head(critic_b.f1_body(sa))
            f2 = critic_b.f2_head(critic_b.f2_body(sa))
            f_list.append(((f1 + f2) / 2.0).cpu().numpy())
    f_vals = np.concatenate(f_list, axis=0).reshape(grid_size, grid_size)
    return f_vals

if __name__ == '__main__':
    all_logs = {}
    for cond in CONDITIONS:
        all_logs[cond] = {}
        for seed in SEEDS:
            path = os.path.join(data_dir, 'log_' + cond + '_' + str(seed) + '.csv')
            all_logs[cond][seed] = pd.read_csv(path)
    eval_results = {}
    for cond in CONDITIONS:
        eval_results[cond] = {}
        for seed in SEEDS:
            actor = load_actor(cond, seed, DEVICE)
            rets, ups = evaluate_policy(actor, seed, EVAL_EPISODES, DEVICE)
            eval_results[cond][seed] = {'returns': rets, 'uprights': ups, 'mean_return': float(np.mean(rets)), 'std_return': float(np.std(rets)), 'mean_upright': float(np.mean(ups)), 'std_upright': float(np.std(ups))}
    eval_rows = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            r = eval_results[cond][seed]
            eval_rows.append({'condition': cond, 'seed': seed, 'mean_return': r['mean_return'], 'std_return': r['std_return'], 'mean_upright': r['mean_upright'], 'std_upright': r['std_upright']})
    pd.DataFrame(eval_rows).to_csv(os.path.join(data_dir, 'eval_results.csv'), index=False)
    sample_eff = {}
    for cond in CONDITIONS:
        se_vals = []
        for seed in SEEDS:
            se = compute_sample_efficiency(all_logs[cond][seed])
            se_vals.append(se)
        sample_eff[cond] = se_vals
    se_rows = []
    for cond in CONDITIONS:
        for i, seed in enumerate(SEEDS):
            se_rows.append({'condition': cond, 'seed': seed, 'steps_to_90pct': sample_eff[cond][i]})
    pd.DataFrame(se_rows).to_csv(os.path.join(data_dir, 'sample_efficiency.csv'), index=False)
    median_seeds = {}
    for cond in CONDITIONS:
        sorted_seeds = sorted(SEEDS, key=lambda s: eval_results[cond][s]['mean_return'])
        median_seeds[cond] = sorted_seeds[len(sorted_seeds) // 2]
    states_np, phi_grid, theta_vals, thetadot_vals, TH, TD = build_state_grid(GRID_SIZE)
    q_grids = {}
    for cond in CONDITIONS:
        mseed = median_seeds[cond]
        critic = load_critic(cond, mseed, DEVICE)
        q_grids[cond] = eval_q_grid(critic, states_np, GRID_SIZE, DEVICE)
    critic_b_rep = load_critic('B', median_seeds['B'], DEVICE)
    f_grid = get_residual_f_grid(critic_b_rep, states_np, GRID_SIZE, DEVICE)
    mse_A = float(np.mean((q_grids['A'] - phi_grid) ** 2))
    mse_B = float(np.mean((q_grids['B'] - phi_grid) ** 2))
    np.save(os.path.join(data_dir, 'phi_grid.npy'), phi_grid)
    np.save(os.path.join(data_dir, 'q_grid_A.npy'), q_grids['A'])
    np.save(os.path.join(data_dir, 'q_grid_B.npy'), q_grids['B'])
    np.save(os.path.join(data_dir, 'f_grid_B.npy'), f_grid)
    np.save(os.path.join(data_dir, 'theta_vals.npy'), theta_vals)
    np.save(os.path.join(data_dir, 'thetadot_vals.npy'), thetadot_vals)
    pd.DataFrame([{'condition': 'A', 'mse': mse_A}, {'condition': 'B', 'mse': mse_B}]).to_csv(os.path.join(data_dir, 'mse_results.csv'), index=False)
    print('Evaluation complete. MSE A:', mse_A, 'MSE B:', mse_B)