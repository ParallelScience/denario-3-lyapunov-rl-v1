# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pickle
import torch
from scipy import stats
import pandas as pd
import gymnasium as gym
from step_1 import SAC, SACArgs, compute_phi_tensor

def main():
    with open(os.path.join('data', 'step_2_results.pkl'), 'rb') as f:
        results_2 = pickle.load(f)
    with open(os.path.join('data', 'step_3_results.pkl'), 'rb') as f:
        results_3 = pickle.load(f)
    all_results = results_2 + results_3
    conditions = ['vanilla', 'condition_A', 'condition_B']
    grouped_results = {cond: [] for cond in conditions}
    for res in all_results:
        grouped_results[res['condition']].append(res)
    max_steps = 30000
    common_steps = np.linspace(0, max_steps, 300)
    learning_curves = {}
    for cond in conditions:
        interp_rewards_list = []
        for res in grouped_results[cond]:
            steps = np.array(res['episode_end_steps'])
            rewards = np.array(res['episode_rewards'])
            if len(rewards) == 0:
                continue
            smoothed_rewards = pd.Series(rewards).rolling(window=10, min_periods=1).mean().values
            interp_rewards = np.interp(common_steps, steps, smoothed_rewards)
            interp_rewards_list.append(interp_rewards)
        if interp_rewards_list:
            interp_rewards_matrix = np.vstack(interp_rewards_list)
            mean_curve = np.mean(interp_rewards_matrix, axis=0)
            std_curve = np.std(interp_rewards_matrix, axis=0)
            learning_curves[cond] = {'mean': mean_curve, 'std': std_curve, 'all': interp_rewards_matrix}
    global_max_reward = -np.inf
    for cond in conditions:
        if cond in learning_curves:
            global_max_reward = max(global_max_reward, np.max(learning_curves[cond]['mean']))
    threshold = 0.9 * global_max_reward
    metrics = {}
    for cond in conditions:
        metrics[cond] = {'sample_efficiency': [], 'final_stability': [], 'final_reward': []}
        for i, res in enumerate(grouped_results[cond]):
            if cond in learning_curves:
                curve = learning_curves[cond]['all'][i]
                idx = np.where(curve >= threshold)[0]
                se = common_steps[idx[0]] if len(idx) > 0 else max_steps
                metrics[cond]['sample_efficiency'].append(se)
            eval_steps = np.array(res['eval_steps'])
            eval_metrics = np.array(res['eval_metrics'])
            valid_evals = eval_metrics[eval_steps >= 20000]
            metrics[cond]['final_stability'].append(np.mean(valid_evals) if len(valid_evals) > 0 else 0.0)
            rewards = res['episode_rewards']
            if len(rewards) >= 10:
                metrics[cond]['final_reward'].append(np.mean(rewards[-10:]))
            elif len(rewards) > 0:
                metrics[cond]['final_reward'].append(np.mean(rewards))
            else:
                metrics[cond]['final_reward'].append(0.0)
    print('--- Performance Metrics ---')
    for cond in conditions:
        se_mean = np.mean(metrics[cond]['sample_efficiency'])
        se_std = np.std(metrics[cond]['sample_efficiency'])
        stab_mean = np.mean(metrics[cond]['final_stability'])
        stab_std = np.std(metrics[cond]['final_stability'])
        rew_mean = np.mean(metrics[cond]['final_reward'])
        rew_std = np.std(metrics[cond]['final_reward'])
        print(cond + ':')
        print('  Sample Efficiency (steps to 90% max reward): ' + str(round(se_mean, 2)) + ' +/- ' + str(round(se_std, 2)))
        print('  Final Upright Stability: ' + str(round(stab_mean, 4)) + ' +/- ' + str(round(stab_std, 4)))
        print('  Final Average Reward: ' + str(round(rew_mean, 2)) + ' +/- ' + str(round(rew_std, 2)))
    print('\n--- Statistical Tests (Condition A vs Condition B) ---')
    se_A = metrics['condition_A']['sample_efficiency']
    se_B = metrics['condition_B']['sample_efficiency']
    t_stat_se, p_val_se = stats.ttest_ind(se_A, se_B)
    print('Sample Efficiency t-test: t=' + str(round(t_stat_se, 4)) + ', p=' + str(round(p_val_se, 4)))
    stab_A = metrics['condition_A']['final_stability']
    stab_B = metrics['condition_B']['final_stability']
    t_stat_stab, p_val_stab = stats.ttest_ind(stab_A, stab_B)
    print('Final Stability t-test: t=' + str(round(t_stat_stab, 4)) + ', p=' + str(round(p_val_stab, 4)))
    rew_A = metrics['condition_A']['final_reward']
    rew_B = metrics['condition_B']['final_reward']
    t_stat_rew, p_val_rew = stats.ttest_ind(rew_A, rew_B)
    print('Final Reward t-test: t=' + str(round(t_stat_rew, 4)) + ', p=' + str(round(p_val_rew, 4)))
    theta_vals = np.linspace(-np.pi, np.pi, 100)
    dot_theta_vals = np.linspace(-8, 8, 100)
    Theta, DotTheta = np.meshgrid(theta_vals, dot_theta_vals)
    states_grid = np.stack([np.cos(Theta).flatten(), np.sin(Theta).flatten(), DotTheta.flatten()], axis=1)
    sorted_indices_A = np.argsort(metrics['condition_A']['final_reward'])
    median_seed_A = grouped_results['condition_A'][sorted_indices_A[len(sorted_indices_A)//2]]['seed']
    sorted_indices_B = np.argsort(metrics['condition_B']['final_reward'])
    median_seed_B = grouped_results['condition_B'][sorted_indices_B[len(sorted_indices_B)//2]]['seed']
    env = gym.make('Pendulum-v1')
    args_A = SACArgs()
    args_A.structured = False
    agent_A = SAC(env.observation_space.shape[0], env.action_space, args_A)
    checkpoint_A = torch.load(os.path.join('data', 'model_condition_A_seed_' + str(median_seed_A) + '.pt'), map_location=agent_A.device)
    agent_A.critic.load_state_dict(checkpoint_A['critic'])
    agent_A.policy.load_state_dict(checkpoint_A['policy'])
    args_B = SACArgs()
    args_B.structured = True
    agent_B = SAC(env.observation_space.shape[0], env.action_space, args_B)
    checkpoint_B = torch.load(os.path.join('data', 'model_condition_B_seed_' + str(median_seed_B) + '.pt'), map_location=agent_B.device)
    agent_B.critic.load_state_dict(checkpoint_B['critic'])
    agent_B.policy.load_state_dict(checkpoint_B['policy'])
    states_tensor = torch.FloatTensor(states_grid).to(agent_A.device)
    phi_grid = compute_phi_tensor(states_tensor).cpu().numpy().reshape(100, 100)
    with torch.no_grad():
        _, _, action_A = agent_A.policy.sample(states_tensor)
        q_A_grid = agent_A.critic(states_tensor, action_A)[0].cpu().numpy().reshape(100, 100)
        _, _, action_B = agent_B.policy.sample(states_tensor)
        q_B_grid = agent_B.critic(states_tensor, action_B)[0].cpu().numpy().reshape(100, 100)
        residual_B_grid = q_B_grid - phi_grid
    np.savez(os.path.join('data', 'grid_data.npz'), theta=theta_vals, dot_theta=dot_theta_vals, phi=phi_grid, q_A=q_A_grid, q_B=q_B_grid, residual_B=residual_B_grid)
    with open(os.path.join('data', 'computed_metrics.pkl'), 'wb') as f:
        pickle.dump({'common_steps': common_steps, 'learning_curves': learning_curves, 'metrics': metrics}, f)

if __name__ == '__main__':
    main()