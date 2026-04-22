# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import torch
import gymnasium as gym
from step_1 import SAC, SACArgs, compute_phi_tensor

def main():
    print('Loading training logs...')
    with open(os.path.join('data', 'step_2_logs.pkl'), 'rb') as f:
        logs_2 = pickle.load(f)
    with open(os.path.join('data', 'step_3_logs.pkl'), 'rb') as f:
        logs_3 = pickle.load(f)
    all_logs = {**logs_2, **logs_3}
    conditions = ['vanilla', 'condition_A', 'condition_B']
    step_grid = np.linspace(0, 100000, 1000)
    learning_curves = {cond: [] for cond in conditions}
    final_rewards = {cond: [] for cond in conditions}
    final_upright = {cond: [] for cond in conditions}
    for cond in conditions:
        for seed in range(5):
            logs = all_logs[cond][seed]
            cum_steps = np.cumsum(logs['episode_lengths'])
            rewards = logs['episode_rewards']
            smoothed_rewards = pd.Series(rewards).rolling(10, min_periods=1).mean().values
            interp_rewards = np.interp(step_grid, cum_steps, smoothed_rewards)
            learning_curves[cond].append(interp_rewards)
            final_rewards[cond].append(np.mean(rewards[-10:]))
            final_upright[cond].append(logs['eval_upright_fractions'][-1])
    mean_curves = {cond: np.mean(learning_curves[cond], axis=0) for cond in conditions}
    std_curves = {cond: np.std(learning_curves[cond], axis=0) for cond in conditions}
    max_lyap = max(np.max(mean_curves['condition_A']), np.max(mean_curves['condition_B']))
    threshold_lyap = 0.9 * max_lyap
    max_vanilla = np.max(mean_curves['vanilla'])
    min_vanilla = np.min(mean_curves['vanilla'])
    threshold_vanilla = max_vanilla - 0.1 * (max_vanilla - min_vanilla)
    sample_efficiency = {cond: [] for cond in conditions}
    for cond in conditions:
        thresh = threshold_vanilla if cond == 'vanilla' else threshold_lyap
        for seed in range(5):
            curve = learning_curves[cond][seed]
            idx = np.where(curve >= thresh)[0]
            if len(idx) > 0:
                sample_efficiency[cond].append(step_grid[idx[0]])
            else:
                sample_efficiency[cond].append(100000.0)
    print('\n' + '='*50)
    print('PERFORMANCE METRICS SUMMARY')
    print('='*50)
    for cond in conditions:
        print('\nCondition: ' + cond)
        print('  Final Reward (last 10 eps): ' + str(np.mean(final_rewards[cond])) + ' +/- ' + str(np.std(final_rewards[cond])))
        print('  Final Upright Fraction:     ' + str(np.mean(final_upright[cond])) + ' +/- ' + str(np.std(final_upright[cond])))
        thresh_print = threshold_vanilla if cond == 'vanilla' else threshold_lyap
        print('  Sample Efficiency (steps to ' + str(thresh_print) + '): ' + str(np.mean(sample_efficiency[cond])) + ' +/- ' + str(np.std(sample_efficiency[cond])))
    print('\n' + '='*50)
    print('STATISTICAL TESTS (Condition A vs Condition B)')
    print('='*50)
    t_stat_perf, p_val_perf = stats.ttest_ind(final_rewards['condition_A'], final_rewards['condition_B'])
    print('Final Performance t-test:      t = ' + str(t_stat_perf) + ', p = ' + str(p_val_perf))
    t_stat_eff, p_val_eff = stats.ttest_ind(sample_efficiency['condition_A'], sample_efficiency['condition_B'])
    print('Sample Efficiency t-test:      t = ' + str(t_stat_eff) + ', p = ' + str(p_val_eff))
    t_stat_upright, p_val_upright = stats.ttest_ind(final_upright['condition_A'], final_upright['condition_B'])
    print('Final Upright Stability t-test: t = ' + str(t_stat_upright) + ', p = ' + str(p_val_upright))
    median_seed_A = np.argsort(final_rewards['condition_A'])[2]
    median_seed_B = np.argsort(final_rewards['condition_B'])[2]
    print('\n' + '='*50)
    print('MEDIAN SEED SELECTION')
    print('='*50)
    print('Median seed for Condition A: ' + str(median_seed_A) + ' (Reward: ' + str(final_rewards['condition_A'][median_seed_A]) + ')')
    print('Median seed for Condition B: ' + str(median_seed_B) + ' (Reward: ' + str(final_rewards['condition_B'][median_seed_B]) + ')')
    print('\nGenerating 2D state grid and computing value functions...')
    theta_vals = np.linspace(-np.pi, np.pi, 100)
    dot_theta_vals = np.linspace(-8, 8, 100)
    THETA, DOT_THETA = np.meshgrid(theta_vals, dot_theta_vals, indexing='ij')
    cos_theta = np.cos(THETA)
    sin_theta = np.sin(THETA)
    grid_states = np.stack([cos_theta.flatten(), sin_theta.flatten(), DOT_THETA.flatten()], axis=1)
    env = gym.make('Pendulum-v1')
    args_A = SACArgs()
    args_A.structured = False
    agent_A = SAC(3, env.action_space, args_A)
    agent_A.critic.load_state_dict(torch.load(os.path.join('data', 'condition_A_critic_seed_' + str(median_seed_A) + '.pth'), map_location=agent_A.device))
    agent_A.policy.load_state_dict(torch.load(os.path.join('data', 'condition_A_policy_seed_' + str(median_seed_A) + '.pth'), map_location=agent_A.device))
    args_B = SACArgs()
    args_B.structured = True
    agent_B = SAC(3, env.action_space, args_B)
    agent_B.critic.load_state_dict(torch.load(os.path.join('data', 'condition_B_critic_seed_' + str(median_seed_B) + '.pth'), map_location=agent_B.device))
    agent_B.policy.load_state_dict(torch.load(os.path.join('data', 'condition_B_policy_seed_' + str(median_seed_B) + '.pth'), map_location=agent_B.device))
    grid_states_tensor = torch.FloatTensor(grid_states).to(agent_A.device)
    with torch.no_grad():
        phi_tensor = compute_phi_tensor(grid_states_tensor)
        phi_grid = phi_tensor.cpu().numpy().reshape(100, 100)
        _, _, mean_action_A = agent_A.policy.sample(grid_states_tensor)
        q1_A, q2_A = agent_A.critic(grid_states_tensor, mean_action_A)
        q_A_grid = torch.min(q1_A, q2_A).cpu().numpy().reshape(100, 100)
        _, _, mean_action_B = agent_B.policy.sample(grid_states_tensor)
        q1_B, q2_B = agent_B.critic(grid_states_tensor, mean_action_B)
        q_B_grid = torch.min(q1_B, q2_B).cpu().numpy().reshape(100, 100)
        residual_B_grid = q_B_grid - phi_grid
    metrics = {'step_grid': step_grid, 'mean_curves': mean_curves, 'std_curves': std_curves, 'final_rewards': final_rewards, 'final_upright': final_upright, 'sample_efficiency': sample_efficiency, 't_stat_perf': t_stat_perf, 'p_val_perf': p_val_perf, 't_stat_eff': t_stat_eff, 'p_val_eff': p_val_eff, 'median_seed_A': median_seed_A, 'median_seed_B': median_seed_B}
    metrics_path = os.path.join('data', 'computed_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    grid_path = os.path.join('data', 'grid_data.npz')
    np.savez(grid_path, theta=THETA, dot_theta=DOT_THETA, phi=phi_grid, q_A=q_A_grid, q_B=q_B_grid, residual_B=residual_B_grid)
    print('\nMetrics successfully saved to ' + metrics_path)
    print('Grid data successfully saved to ' + grid_path)

if __name__ == '__main__':
    main()