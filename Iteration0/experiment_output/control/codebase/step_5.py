# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_learning_and_stability(metrics, timestamp):
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    step_grid = metrics['step_grid']
    eval_steps = metrics['eval_steps']
    colors = {'vanilla': 'gray', 'condition_A': 'blue', 'condition_B': 'orange'}
    labels = {'vanilla': 'Vanilla SAC', 'condition_A': 'Condition A (Direct)', 'condition_B': 'Condition B (Structured)'}
    axins = ax1.inset_axes([0.4, 0.15, 0.55, 0.4])
    for cond in ['vanilla', 'condition_A', 'condition_B']:
        lc = metrics['learning_curves'][cond]
        mean_lc = np.mean(lc, axis=0)
        std_lc = np.std(lc, axis=0)
        ax1.plot(step_grid, mean_lc, label=labels[cond], color=colors[cond])
        ax1.fill_between(step_grid, mean_lc - std_lc, mean_lc + std_lc, color=colors[cond], alpha=0.2)
        axins.plot(step_grid, mean_lc, color=colors[cond])
        axins.fill_between(step_grid, mean_lc - std_lc, mean_lc + std_lc, color=colors[cond], alpha=0.2)
        stab = metrics['eval_stabilities'][cond]
        mean_stab = np.mean(stab, axis=0)
        std_stab = np.std(stab, axis=0)
        ax2.plot(eval_steps, mean_stab, label=labels[cond], color=colors[cond], marker='o')
        ax2.fill_between(eval_steps, np.clip(mean_stab - std_stab, 0, 1), np.clip(mean_stab + std_stab, 0, 1), color=colors[cond], alpha=0.2)
    ax1.set_title('Learning Curves')
    ax1.set_xlabel('Environment Steps')
    ax1.set_ylabel('Cumulative Reward')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    axins.set_ylim(-50, 10)
    axins.set_xlim(0, 100000)
    axins.set_title('Zoomed In [-50, 10]', fontsize=10)
    axins.grid(True)
    ax1.indicate_inset_zoom(axins, edgecolor='black')
    ax2.set_title('Upright Stability Over Time')
    ax2.set_xlabel('Environment Steps')
    ax2.set_ylabel('Fraction of Time Upright (|theta| < 0.1 rad)')
    ax2.legend()
    ax2.grid(True)
    fig1.tight_layout()
    fig1_path = os.path.join('data', 'learning_and_stability_1_' + str(timestamp) + '.png')
    fig1.savefig(fig1_path, dpi=300)
    print('Plot saved to ' + fig1_path)
    plt.close(fig1)

def plot_value_function_heatmaps(grid_data, timestamp):
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    theta = grid_data['theta']
    theta_dot = grid_data['theta_dot']
    phi = grid_data['phi']
    q_A = grid_data['q_A']
    q_B = grid_data['q_B']
    residual_B = grid_data['residual_B']
    vmin = min(np.min(phi), np.min(q_A), np.min(q_B))
    vmax = max(np.max(phi), np.max(q_A), np.max(q_B))
    im0 = axes[0, 0].pcolormesh(theta, theta_dot, phi, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('(a) Analytic Lyapunov Function Phi(s)')
    axes[0, 0].set_xlabel('Angle theta (rad)')
    axes[0, 0].set_ylabel('Angular Velocity theta_dot (rad/s)')
    fig2.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].pcolormesh(theta, theta_dot, q_A, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('(b) Condition A Learned Value V(s)')
    axes[0, 1].set_xlabel('Angle theta (rad)')
    axes[0, 1].set_ylabel('Angular Velocity theta_dot (rad/s)')
    fig2.colorbar(im1, ax=axes[0, 1])
    im2 = axes[1, 0].pcolormesh(theta, theta_dot, q_B, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('(c) Condition B Learned Value V(s)')
    axes[1, 0].set_xlabel('Angle theta (rad)')
    axes[1, 0].set_ylabel('Angular Velocity theta_dot (rad/s)')
    fig2.colorbar(im2, ax=axes[1, 0])
    res_max = np.max(np.abs(residual_B))
    norm = mcolors.TwoSlopeNorm(vcenter=0., vmin=-res_max, vmax=res_max)
    im3 = axes[1, 1].pcolormesh(theta, theta_dot, residual_B, shading='auto', cmap='RdBu_r', norm=norm)
    axes[1, 1].set_title('(d) Condition B Learned Residual f_theta(s)')
    axes[1, 1].set_xlabel('Angle theta (rad)')
    axes[1, 1].set_ylabel('Angular Velocity theta_dot (rad/s)')
    fig2.colorbar(im3, ax=axes[1, 1])
    fig2.tight_layout()
    fig2_path = os.path.join('data', 'value_function_heatmaps_2_' + str(timestamp) + '.png')
    fig2.savefig(fig2_path, dpi=300)
    print('Plot saved to ' + fig2_path)
    plt.close(fig2)

if __name__ == '__main__':
    with open(os.path.join('data', 'computed_metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
    grid_data = np.load(os.path.join('data', 'grid_data.npz'))
    timestamp = int(time.time())
    plot_learning_and_stability(metrics, timestamp)
    plot_value_function_heatmaps(grid_data, timestamp)