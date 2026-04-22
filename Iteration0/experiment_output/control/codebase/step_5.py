# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams['text.usetex'] = False
def main():
    with open(os.path.join('data', 'computed_metrics.pkl'), 'rb') as f:
        metrics_data = pickle.load(f)
    common_steps = metrics_data['common_steps']
    learning_curves = metrics_data['learning_curves']
    with open(os.path.join('data', 'step_2_results.pkl'), 'rb') as f:
        results_2 = pickle.load(f)
    with open(os.path.join('data', 'step_3_results.pkl'), 'rb') as f:
        results_3 = pickle.load(f)
    all_results = results_2 + results_3
    conditions = ['condition_A', 'condition_B']
    labels = {'condition_A': 'Condition A (Direct)', 'condition_B': 'Condition B (Structured)'}
    colors = {'condition_A': 'blue', 'condition_B': 'orange'}
    eval_data = {cond: [] for cond in conditions}
    for res in all_results:
        cond = res['condition']
        if cond in conditions:
            eval_data[cond].append((res['eval_steps'], res['eval_metrics']))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for cond in conditions:
        if cond in learning_curves:
            mean_reward = learning_curves[cond]['mean']
            std_reward = learning_curves[cond]['std']
            ax1.plot(common_steps, mean_reward, label=labels[cond], color=colors[cond])
            ax1.fill_between(common_steps, mean_reward - std_reward, mean_reward + std_reward, color=colors[cond], alpha=0.2)
    ax1.set_title('Learning Curves')
    ax1.set_xlabel('Environment Steps')
    ax1.set_ylabel('Episodic Return')
    ax1.legend()
    ax1.grid(True)
    for cond in conditions:
        if eval_data[cond]:
            steps = np.array(eval_data[cond][0][0])
            metrics = np.array([ed[1] for ed in eval_data[cond]])
            mean_metrics = np.mean(metrics, axis=0)
            std_metrics = np.std(metrics, axis=0)
            ax2.plot(steps, mean_metrics, label=labels[cond], color=colors[cond], marker='o')
            ax2.fill_between(steps, mean_metrics - std_metrics, mean_metrics + std_metrics, color=colors[cond], alpha=0.2)
    ax2.set_title('Upright Stability Over Time')
    ax2.set_xlabel('Environment Steps')
    ax2.set_ylabel('Fraction of Time Upright (|theta| < 0.1 rad)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    timestamp = int(time.time())
    plot1_path = os.path.join('data', 'learning_curves_stability_1_' + str(timestamp) + '.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print('Plot saved to ' + plot1_path)
    grid_data = np.load(os.path.join('data', 'grid_data.npz'))
    theta = grid_data['theta']
    dot_theta = grid_data['dot_theta']
    phi = grid_data['phi']
    q_A = grid_data['q_A']
    q_B = grid_data['q_B']
    residual_B = grid_data['residual_B']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(right=0.88, wspace=0.2, hspace=0.25)
    vmin = min(phi.min(), q_A.min(), q_B.min())
    vmax = max(phi.max(), q_A.max(), q_B.max())
    im0 = axes[0, 0].pcolormesh(theta, dot_theta, phi, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('(a) Analytic Lyapunov Function Phi(s)')
    axes[0, 0].set_xlabel('Angle theta (rad)')
    axes[0, 0].set_ylabel('Angular Velocity theta_dot (rad/s)')
    im1 = axes[0, 1].pcolormesh(theta, dot_theta, q_A, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('(b) Condition A Learned Value Function V(s)')
    axes[0, 1].set_xlabel('Angle theta (rad)')
    axes[0, 1].set_ylabel('Angular Velocity theta_dot (rad/s)')
    im2 = axes[1, 0].pcolormesh(theta, dot_theta, q_B, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('(c) Condition B Learned Value Function V(s)')
    axes[1, 0].set_xlabel('Angle theta (rad)')
    axes[1, 0].set_ylabel('Angular Velocity theta_dot (rad/s)')
    cbar_ax_top = fig.add_axes([0.90, 0.55, 0.02, 0.33])
    fig.colorbar(im0, cax=cbar_ax_top, label='Value')
    res_max = max(abs(residual_B.min()), abs(residual_B.max()))
    im3 = axes[1, 1].pcolormesh(theta, dot_theta, residual_B, shading='auto', cmap='RdBu_r', vmin=-res_max, vmax=res_max)
    axes[1, 1].set_title('(d) Condition B Learned Residual f_theta(s)')
    axes[1, 1].set_xlabel('Angle theta (rad)')
    axes[1, 1].set_ylabel('Angular Velocity theta_dot (rad/s)')
    cbar_ax_bottom = fig.add_axes([0.90, 0.12, 0.02, 0.33])
    fig.colorbar(im3, cax=cbar_ax_bottom, label='Residual Value')
    plot2_path = os.path.join('data', 'value_function_heatmaps_2_' + str(timestamp) + '.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print('Plot saved to ' + plot2_path)
if __name__ == '__main__':
    main()