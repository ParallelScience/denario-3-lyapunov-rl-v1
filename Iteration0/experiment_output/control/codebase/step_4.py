# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import pickle
import numpy as np
import torch
import scipy.stats as stats
import gymnasium as gym
from step_1 import SAC, SACArgs, compute_phi_tensor

def safe_ttest(a, b):
    if np.all(a == b):
        return 0.0, 1.0
    elif np.var(a) == 0 and np.var(b) == 0:
        return 0.0, 1.0
    else:
        return stats.ttest_ind(a, b, equal_var=False)

def main():
    with open('data/step_2_logs.pkl', 'rb') as f:
        logs_2 = pickle.load(f)
    with open('data/step_3_logs.pkl', 'rb') as f:
        logs_3 = pickle.load(f)
    logs = {'vanilla': logs_2['vanilla'], 'condition_A': logs_2['condition_A'], 'condition_B': logs_3['condition_B']}
    step_grid = np.linspace(0, 100000, 200)
    learning_curves = {}
    final_performances = {}
    final_stabilities = {}
    for condition in ['vanilla', 'condition_A', 'condition_B']:
        interp_rewards = []
        final_perfs = []
        final_stabs = []
        for seed in range(5):
            ep_lengths = logs[condition][seed]['episode_lengths']
            ep_rewards = logs[condition][seed]['episode_rewards']
            cum_steps = np.cumsum(ep_lengths)
            interp_rew = np.interp(step_grid, cum_steps, ep_rewards)
            interp_rewards.append(interp_rew)
            final_perf = np.mean(ep_rewards[-10:]) if len(ep_rewards) >= 10 else np.mean(ep_rewards)
            final_perfs.append(final_perf)
            final_stab = logs[condition][seed]['eval_upright_fractions'][-1]
            final_stabs.append(final_stab)
        learning_curves[condition] = np.array(interp_rewards)
        final_performances[condition] = np.array(final_perfs)
        final_stabilities[condition] = np.array(final_stabs)
    mean_curve_A = np.mean(learning_curves['condition_A'], axis=0)
    mean_curve_B = np.mean(learning_curves['condition_B'], axis=0)
    max_reward = max(np.max(mean_curve_A), np.max(mean_curve_B))
    threshold = 0.9 * max_reward
    sample_efficiencies = {}
    for condition in ['condition_A', 'condition_B']:
        effs = []
        for seed in range(5):
            curve = learning_curves[condition][seed]
            idx = np.where(curve >= threshold)[0]
            if len(idx) > 0:
                effs.append(step_grid[idx[0]])
            else:
                effs.append(100000.0)
        sample_efficiencies[condition] = np.array(effs)
    t_stat_perf, p_val_perf = safe_ttest(final_performances['condition_A'], final_performances['condition_B'])
    t_stat_eff, p_val_eff = safe_ttest(sample_efficiencies['condition_A'], sample_efficiencies['condition_B'])
    t_stat_stab, p_val_stab = safe_ttest(final_stabilities['condition_A'], final_stabilities['condition_B'])
    print('=== Performance Metrics ===')
    for cond in ['vanilla', 'condition_A', 'condition_B']:
        print('Condition: ' + cond)
        print('  Final Performance (Avg last 10 eps): ' + str(round(np.mean(final_performances[cond]), 2)) + ' +/- ' + str(round(np.std(final_performances[cond]), 2)))
        print('  Final Upright Stability: ' + str(round(np.mean(final_stabilities[cond]), 4)) + ' +/- ' + str(round(np.std(final_stabilities[cond]), 4)))
        if cond in sample_efficiencies:
            print('  Sample Efficiency (steps to 90% max reward): ' + str(round(np.mean(sample_efficiencies[cond]))) + ' +/- ' + str(round(np.std(sample_efficiencies[cond]))))
    print('\n=== Statistical Tests (Condition A vs Condition B) ===')
    print('Final Performance: t-stat = ' + str(round(t_stat_perf, 4)) + ', p-value = ' + str(round(p_val_perf, 4)))
    print('Sample Efficiency: t-stat = ' + str(round(t_stat_eff, 4)) + ', p-value = ' + str(round(p_val_eff, 4)))
    print('Final Stability:   t-stat = ' + str(round(t_stat_stab, 4)) + ', p-value = ' + str(round(p_val_stab, 4)))
    median_seed_A = int(np.argsort(final_performances['condition_A'])[2])
    median_seed_B = int(np.argsort(final_performances['condition_B'])[2])
    theta_vals = np.linspace(-np.pi, np.pi, 100)
    theta_dot_vals = np.linspace(-8, 8, 100)
    THETA, THETA_DOT = np.meshgrid(theta_vals, theta_dot_vals)
    states = np.stack([np.cos(THETA).flatten(), np.sin(THETA).flatten(), THETA_DOT.flatten()], axis=1)
    env = gym.make('Pendulum-v1')
    args_A = SACArgs()
    args_A.structured = False
    agent_A = SAC(env.observation_space.shape[0], env.action_space, args_A)
    agent_A.critic.load_state_dict(torch.load(os.path.join('data', 'condition_A_critic_seed_' + str(median_seed_A) + '.pth'), map_location=agent_A.device))
    agent_A.policy.load_state_dict(torch.load(os.path.join('data', 'condition_A_policy_seed_' + str(median_seed_A) + '.pth'), map_location=agent_A.device))
    args_B = SACArgs()
    args_B.structured = True
    agent_B = SAC(env.observation_space.shape[0], env.action_space, args_B)
    agent_B.critic.load_state_dict(torch.load(os.path.join('data', 'condition_B_critic_seed_' + str(median_seed_B) + '.pth'), map_location=agent_B.device))
    agent_B.policy.load_state_dict(torch.load(os.path.join('data', 'condition_B_policy_seed_' + str(median_seed_B) + '.pth'), map_location=agent_B.device))
    states_tensor = torch.FloatTensor(states).to(agent_A.device)
    with torch.no_grad():
        mean_A, _ = agent_A.policy(states_tensor)
        action_A = torch.tanh(mean_A) * agent_A.policy.action_scale + agent_A.policy.action_bias
        q1_A, q2_A = agent_A.critic(states_tensor, action_A)
        q_A = torch.min(q1_A, q2_A).cpu().numpy().reshape(100, 100)
        mean_B, _ = agent_B.policy(states_tensor)
        action_B = torch.tanh(mean_B) * agent_B.policy.action_scale + agent_B.policy.action_bias
        q1_B, q2_B = agent_B.critic(states_tensor, action_B)
        q_B = torch.min(q1_B, q2_B).cpu().numpy().reshape(100, 100)
        phi = compute_phi_tensor(states_tensor).cpu().numpy().reshape(100, 100)
        residual_B = q_B - phi
    eval_steps = logs['vanilla'][0]['eval_steps']
    eval_stabilities = {}
    for condition in ['vanilla', 'condition_A', 'condition_B']:
        stabs = []
        for seed in range(5):
            stabs.append(logs[condition][seed]['eval_upright_fractions'])
        eval_stabilities[condition] = np.array(stabs)
    metrics = {'step_grid': step_grid, 'learning_curves': learning_curves, 'eval_steps': eval_steps, 'eval_stabilities': eval_stabilities, 'final_performances': final_performances, 'final_stabilities': final_stabilities, 'sample_efficiencies': sample_efficiencies, 't_tests': {'performance': (t_stat_perf, p_val_perf), 'efficiency': (t_stat_eff, p_val_eff), 'stability': (t_stat_stab, p_val_stab)}, 'median_seeds': {'condition_A': median_seed_A, 'condition_B': median_seed_B}}
    with open(os.path.join('data', 'computed_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    np.savez(os.path.join('data', 'grid_data.npz'), theta=THETA, theta_dot=THETA_DOT, phi=phi, q_A=q_A, q_B=q_B, residual_B=residual_B)
    print('\nData saved to data/computed_metrics.pkl and data/grid_data.npz')

if __name__ == '__main__':
    main()