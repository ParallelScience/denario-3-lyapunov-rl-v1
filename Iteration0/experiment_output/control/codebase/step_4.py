# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import pickle
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
import gymnasium as gym
from step_1 import SAC, SACArgs, compute_phi_tensor

def safe_ttest(a, b):
    if np.std(a) == 0 and np.std(b) == 0:
        if np.mean(a) == np.mean(b):
            return 0.0, 1.0
        else:
            return float('inf'), 0.0
    res = ttest_ind(a, b)
    return res.statistic, res.pvalue

def main():
    data_dir = 'data'
    with open(os.path.join(data_dir, 'step_2_results.pkl'), 'rb') as f:
        results_2 = pickle.load(f)
    with open(os.path.join(data_dir, 'step_3_results.pkl'), 'rb') as f:
        results_3 = pickle.load(f)
    all_results = results_2 + results_3
    conditions = ['vanilla', 'condition_A', 'condition_B']
    data_by_cond = {cond: [] for cond in conditions}
    for res in all_results:
        data_by_cond[res['condition']].append(res)
    common_steps = np.linspace(0, 30000, 300)
    interp_rewards = {cond: [] for cond in conditions}
    final_rewards = {cond: [] for cond in conditions}
    final_stability = {cond: [] for cond in conditions}
    for cond in conditions:
        for res in data_by_cond[cond]:
            steps = res['episode_end_steps']
            rewards = res['episode_rewards']
            steps, unique_indices = np.unique(steps, return_index=True)
            rewards = np.array(rewards)[unique_indices]
            if len(steps) == 0:
                continue
            f_interp = interp1d(steps, rewards, kind='linear', bounds_error=False, fill_value=(rewards[0], rewards[-1]))
            interp_rewards[cond].append(f_interp(common_steps))
            final_rewards[cond].append(np.mean(rewards[-10:]))
            final_stability[cond].append(res['eval_metrics'][-1])
    mean_rewards = {cond: np.mean(interp_rewards[cond], axis=0) for cond in conditions}
    std_rewards = {cond: np.std(interp_rewards[cond], axis=0) for cond in conditions}
    eval_steps = data_by_cond['vanilla'][0]['eval_steps']
    eval_metrics_all = {cond: [] for cond in conditions}
    for cond in conditions:
        for res in data_by_cond[cond]:
            eval_metrics_all[cond].append(res['eval_metrics'])
    mean_eval = {cond: np.mean(eval_metrics_all[cond], axis=0) for cond in conditions}
    std_eval = {cond: np.std(eval_metrics_all[cond], axis=0) for cond in conditions}
    global_max_reward = max([np.max(mean_rewards[cond]) for cond in conditions])
    threshold = 0.9 * global_max_reward
    sample_efficiency = {cond: [] for cond in conditions}
    for cond in conditions:
        for rewards_curve in interp_rewards[cond]:
            idx = np.where(rewards_curve >= threshold)[0]
            if len(idx) > 0:
                sample_efficiency[cond].append(common_steps[idx[0]])
            else:
                sample_efficiency[cond].append(30000)
    mean_efficiency = {cond: np.mean(sample_efficiency[cond]) for cond in conditions}
    std_efficiency = {cond: np.std(sample_efficiency[cond]) for cond in conditions}
    mean_final_stability = {cond: np.mean(final_stability[cond]) for cond in conditions}
    std_final_stability = {cond: np.std(final_stability[cond]) for cond in conditions}
    t_stat_perf, p_val_perf = safe_ttest(final_rewards['condition_A'], final_rewards['condition_B'])
    t_stat_eff, p_val_eff = safe_ttest(sample_efficiency['condition_A'], sample_efficiency['condition_B'])
    t_stat_stab, p_val_stab = safe_ttest(final_stability['condition_A'], final_stability['condition_B'])
    print('=== Performance Metrics ===')
    for cond in conditions:
        print('Condition: ' + cond)
        print('  Final Reward (last 10 eps): ' + str(round(np.mean(final_rewards[cond]), 2)) + ' +/- ' + str(round(np.std(final_rewards[cond]), 2)))
        print('  Final Upright Stability: ' + str(round(mean_final_stability[cond], 2)) + ' +/- ' + str(round(std_final_stability[cond], 2)))
        print('  Sample Efficiency (steps to 90% max): ' + str(round(mean_efficiency[cond], 0)) + ' +/- ' + str(round(std_efficiency[cond], 0)))
    print('\n=== Statistical Tests (Condition A vs Condition B) ===')
    print('Final Reward: t-stat = ' + str(round(t_stat_perf, 3)) + ', p-value = ' + str(round(p_val_perf, 4)))
    print('Sample Efficiency: t-stat = ' + str(round(t_stat_eff, 3)) + ', p-value = ' + str(round(p_val_eff, 4)))
    print('Final Stability: t-stat = ' + str(round(t_stat_stab, 3)) + ', p-value = ' + str(round(p_val_stab, 4)))
    theta_grid = np.linspace(-np.pi, np.pi, 100)
    dot_theta_grid = np.linspace(-8, 8, 100)
    Theta, DotTheta = np.meshgrid(theta_grid, dot_theta_grid)
    cos_theta = np.cos(Theta)
    sin_theta = np.sin(Theta)
    states_grid = np.stack([cos_theta, sin_theta, DotTheta], axis=-1)
    states_flat = states_grid.reshape(-1, 3)
    idx_A = np.argsort(final_rewards['condition_A'])[len(final_rewards['condition_A'])//2]
    median_seed_A = data_by_cond['condition_A'][idx_A]['seed']
    idx_B = np.argsort(final_rewards['condition_B'])[len(final_rewards['condition_B'])//2]
    median_seed_B = data_by_cond['condition_B'][idx_B]['seed']
    print('\nMedian-performing seed for Condition A: ' + str(median_seed_A))
    print('Median-performing seed for Condition B: ' + str(median_seed_B))
    env = gym.make('Pendulum-v1')
    args_A = SACArgs()
    args_A.structured = False
    agent_A = SAC(env.observation_space.shape[0], env.action_space, args_A)
    checkpoint_A = torch.load(os.path.join(data_dir, 'model_condition_A_seed_' + str(median_seed_A) + '.pt'), map_location=agent_A.device)
    agent_A.critic.load_state_dict(checkpoint_A['critic'])
    agent_A.policy.load_state_dict(checkpoint_A['policy'])
    args_B = SACArgs()
    args_B.structured = True
    agent_B = SAC(env.observation_space.shape[0], env.action_space, args_B)
    checkpoint_B = torch.load(os.path.join(data_dir, 'model_condition_B_seed_' + str(median_seed_B) + '.pt'), map_location=agent_B.device)
    agent_B.critic.load_state_dict(checkpoint_B['critic'])
    agent_B.policy.load_state_dict(checkpoint_B['policy'])
    states_tensor = torch.FloatTensor(states_flat).to(agent_A.device)
    phi_tensor = compute_phi_tensor(states_tensor)
    phi_grid = phi_tensor.cpu().numpy().reshape(100, 100)
    with torch.no_grad():
        _, _, action_mean_A = agent_A.policy.sample(states_tensor)
        q1_A, _ = agent_A.critic(states_tensor, action_mean_A)
        q_grid_A = q1_A.cpu().numpy().reshape(100, 100)
    with torch.no_grad():
        _, _, action_mean_B = agent_B.policy.sample(states_tensor)
        q1_B, _ = agent_B.critic(states_tensor, action_mean_B)
        q_grid_B = q1_B.cpu().numpy().reshape(100, 100)
        residual_grid_B = q_grid_B - phi_grid
    metrics = {'common_steps': common_steps, 'mean_rewards': mean_rewards, 'std_rewards': std_rewards, 'eval_steps': eval_steps, 'mean_eval': mean_eval, 'std_eval': std_eval, 'sample_efficiency': sample_efficiency, 'final_rewards': final_rewards, 'final_stability': final_stability, 't_stat_perf': t_stat_perf, 'p_val_perf': p_val_perf, 't_stat_eff': t_stat_eff, 'p_val_eff': p_val_eff, 't_stat_stab': t_stat_stab, 'p_val_stab': p_val_stab}
    with open(os.path.join(data_dir, 'computed_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    np.savez(os.path.join(data_dir, 'grid_data.npz'), theta_grid=theta_grid, dot_theta_grid=dot_theta_grid, phi_grid=phi_grid, q_grid_A=q_grid_A, q_grid_B=q_grid_B, residual_grid_B=residual_grid_B)
    print('\nMetrics and grid data successfully saved to data/computed_metrics.pkl and data/grid_data.npz')

if __name__ == '__main__':
    main()