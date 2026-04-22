# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data/')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['text.usetex'] = False

def main():
    data_dir = 'data'
    with open(os.path.join(data_dir, 'computed_metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
    grid_data = np.load(os.path.join(data_dir, 'grid_data.npz'))
    with open(os.path.join(data_dir, 'step_2_logs.pkl'), 'rb') as f:
        logs_2 = pickle.load(f)
    with open(os.path.join(data_dir, 'step_3_logs.pkl'), 'rb') as f:
        logs_3 = pickle.load(f)
    all_logs = {**logs_2, **logs_3}
    conditions = ['vanilla', 'condition_A', 'condition_B']
    labels = {'vanilla': 'Vanilla SAC', 'condition_A': 'Condition A (Direct)', 'condition_B': 'Condition B (Structured)'}
    colors = {'vanilla': 'gray', 'condition_A': 'blue', 'condition_B': 'orange'}
    markers = {'vanilla': 's', 'condition_A': 'o', 'condition_B': '^'}
    linestyles = {'vanilla': ':', 'condition_A': '-', 'condition_B': '--'}
    eval_steps = all_logs['vanilla'][0]['eval_steps']
    upright_fractions = {cond: [] for cond in conditions}
    for cond in conditions:
        for seed in range(5):
            upright_fractions[cond].append(all_logs[cond][seed]['eval_upright_fractions'])
    mean_upright = {cond: np.mean(upright_fractions[cond], axis=0) for cond in conditions}
    std_upright = {cond: np.std(upright_fractions[cond], axis=0) for cond in conditions}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig1 = plt.figure(figsize=(15, 7))
    gs = fig1.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1])
    ax1_top = fig1.add_subplot(gs[0, 0])
    ax1_bot = fig1.add_subplot(gs[1, 0])
    ax2 = fig1.add_subplot(gs[:, 1])
    step_grid = metrics['step_grid']
    for cond in ['condition_A', 'condition_B']:
        mean_curve = metrics['mean_curves'][cond]
        std_curve = metrics['std_curves'][cond]
        ax1_top.plot(step_grid, mean_curve, label=labels[cond], color=colors[cond], linestyle=linestyles[cond])
        ax1_top.fill_between(step_grid, mean_curve - std_curve, mean_curve + std_curve, color=colors[cond], alpha=0.2)
    ax1_top.set_xlabel('Environment Steps')
    ax1_top.set_ylabel('Lyapunov Reward')
    ax1_top.set_title('Learning Curves (Conditions A & B)')
    ax1_top.legend(loc='lower right')
    ax1_top.grid(True)
    cond = 'vanilla'
    mean_curve = metrics['mean_curves'][cond]
    std_curve = metrics['std_curves'][cond]
    ax1_bot.plot(step_grid, mean_curve, label=labels[cond], color=colors[cond], linestyle=linestyles[cond])
    ax1_bot.fill_between(step_grid, mean_curve - std_curve, mean_curve + std_curve, color=colors[cond], alpha=0.2)
    ax1_bot.set_xlabel('Environment Steps')
    ax1_bot.set_ylabel('Native Reward')
    ax1_bot.set_title('Learning Curve (Vanilla SAC)')
    ax1_bot.legend(loc='lower right')
    ax1_bot.grid(True)
    for cond in conditions:
        mean_u = mean_upright[cond]
        std_u = std_upright[cond]
        ax2.plot(eval_steps, mean_u, label=labels[cond], color=colors[cond], marker=markers[cond], linestyle=linestyles[cond], markersize=5)
        ax2.fill_between(eval_steps, mean_u - std_u, mean_u + std_u, color=colors[cond], alpha=0.15)
    ax2.set_title('Upright Stability over Time')
    ax2.set_xlabel('Environment Steps')
    ax2.set_ylabel('Fraction of Time Upright (|θ| < 0.1 rad)')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    fig1.tight_layout()
    fig1_path = os.path.join(data_dir, 'learning_curves_and_stability_1_' + timestamp + '.png')
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print('Plot saved to ' + fig1_path)
    theta = grid_data['theta']
    dot_theta = grid_data['dot_theta']
    phi = grid_data['phi']
    q_A = grid_data['q_A']
    q_B = grid_data['q_B']
    residual_B = grid_data['residual_B']
    fig2, axs = plt.subplots(2, 2, figsize=(14, 10))
    vmin = min(np.min(phi), np.min(q_A), np.min(q_B))
    vmax = max(np.max(phi), np.max(q_A), np.max(q_B))
    im0 = axs[0, 0].pcolormesh(theta, dot_theta, phi, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('(a) Analytic Lyapunov Function Φ(s)')
    axs[0, 0].set_xlabel('Angle θ (rad)')
    axs[0, 0].set_ylabel('Angular Velocity θ̇ (rad/s)')
    fig2.colorbar(im0, ax=axs[0, 0])
    im1 = axs[0, 1].pcolormesh(theta, dot_theta, q_A, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('(b) Condition A Learned Value V(s)')
    axs[0, 1].set_xlabel('Angle θ (rad)')
    axs[0, 1].set_ylabel('Angular Velocity θ̇ (rad/s)')
    fig2.colorbar(im1, ax=axs[0, 1])
    im2 = axs[1, 0].pcolormesh(theta, dot_theta, q_B, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('(c) Condition B Learned Value V(s)')
    axs[1, 0].set_xlabel('Angle θ (rad)')
    axs[1, 0].set_ylabel('Angular Velocity θ̇ (rad/s)')
    fig2.colorbar(im2, ax=axs[1, 0])
    res_max = np.max(np.abs(residual_B))
    im3 = axs[1, 1].pcolormesh(theta, dot_theta, residual_B, shading='auto', cmap='RdBu_r', vmin=-res_max, vmax=res_max)
    axs[1, 1].set_title('(d) Condition B Learned Residual f_θ(s)')
    axs[1, 1].set_xlabel('Angle θ (rad)')
    axs[1, 1].set_ylabel('Angular Velocity θ̇ (rad/s)')
    fig2.colorbar(im3, ax=axs[1, 1])
    fig2.tight_layout()
    fig2_path = os.path.join(data_dir, 'value_function_heatmaps_2_' + timestamp + '.png')
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print('Plot saved to ' + fig2_path)

if __name__ == '__main__':
    main()