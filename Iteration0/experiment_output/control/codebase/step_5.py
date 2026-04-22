# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

def main():
    mpl.rcParams['text.usetex'] = False
    data_dir = 'data'
    with open(os.path.join(data_dir, 'computed_metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
    grid_data = np.load(os.path.join(data_dir, 'grid_data.npz'))
    timestamp = str(int(time.time()))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    conditions = ['condition_A', 'condition_B']
    labels = {'condition_A': 'Condition A (Direct)', 'condition_B': 'Condition B (Structured)'}
    colors = {'condition_A': 'blue', 'condition_B': 'orange'}
    common_steps = metrics['common_steps']
    eval_steps = metrics['eval_steps']
    for cond in conditions:
        mean_r = metrics['mean_rewards'][cond]
        std_r = metrics['std_rewards'][cond]
        ax1.plot(common_steps, mean_r, label=labels[cond], color=colors[cond])
        ax1.fill_between(common_steps, mean_r - std_r, mean_r + std_r, color=colors[cond], alpha=0.2)
        mean_e = metrics['mean_eval'][cond]
        std_e = metrics['std_eval'][cond]
        ax2.plot(eval_steps, mean_e, label=labels[cond], color=colors[cond], marker='o')
        ax2.fill_between(eval_steps, mean_e - std_e, mean_e + std_e, color=colors[cond], alpha=0.2)
    ax1.set_title('Learning Curves: Episode Reward (Mean ± Std, 5 seeds)')
    ax1.set_xlabel('Environment Steps')
    ax1.set_ylabel('Episode Reward')
    ax1.legend()
    ax1.grid(True)
    ax2.set_title('Upright Stability over Time (Mean ± Std, 5 seeds)')
    ax2.set_xlabel('Environment Steps')
    ax2.set_ylabel('Fraction of Time Upright (|theta| < 0.1 rad)')
    ax2.legend()
    ax2.grid(True)
    fig.tight_layout()
    plot1_path = os.path.join(data_dir, 'learning_curves_1_' + timestamp + '.png')
    fig.savefig(plot1_path, dpi=300)
    plt.close(fig)
    print('Plot saved to ' + plot1_path)
    theta_grid = grid_data['theta_grid']
    dot_theta_grid = grid_data['dot_theta_grid']
    phi_grid = grid_data['phi_grid']
    q_grid_A = grid_data['q_grid_A']
    q_grid_B = grid_data['q_grid_B']
    residual_grid_B = grid_data['residual_grid_B']
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    vmin = 0.0
    vmax = 40.0
    extent = [theta_grid.min(), theta_grid.max(), dot_theta_grid.min(), dot_theta_grid.max()]
    im0 = axes[0].imshow(phi_grid, origin='lower', extent=extent, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0].set_title('(a) Analytic Lyapunov Function Phi(s)')
    axes[0].set_xlabel('Angle theta (rad)')
    axes[0].set_ylabel('Angular Velocity dot(theta) (rad/s)')
    fig2.colorbar(im0, ax=axes[0], label='Value')
    im1 = axes[1].imshow(q_grid_A, origin='lower', extent=extent, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1].set_title('(b) Condition A Learned Value V(s)')
    axes[1].set_xlabel('Angle theta (rad)')
    axes[1].set_ylabel('Angular Velocity dot(theta) (rad/s)')
    fig2.colorbar(im1, ax=axes[1], label='Value')
    im2 = axes[2].imshow(q_grid_B, origin='lower', extent=extent, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[2].set_title('(c) Condition B Learned Value V(s)')
    axes[2].set_xlabel('Angle theta (rad)')
    axes[2].set_ylabel('Angular Velocity dot(theta) (rad/s)')
    fig2.colorbar(im2, ax=axes[2], label='Value')
    res_max = float(np.abs(residual_grid_B).max())
    if res_max == 0:
        res_max = 1.0
    im3 = axes[3].imshow(residual_grid_B, origin='lower', extent=extent, aspect='auto', vmin=-res_max, vmax=res_max, cmap='RdBu_r')
    axes[3].set_title('(d) Condition B Learned Residual f_theta(s)')
    axes[3].set_xlabel('Angle theta (rad)')
    axes[3].set_ylabel('Angular Velocity dot(theta) (rad/s)')
    fig2.colorbar(im3, ax=axes[3], label='Residual Value')
    fig2.tight_layout()
    plot2_path = os.path.join(data_dir, 'value_heatmaps_2_' + timestamp + '.png')
    fig2.savefig(plot2_path, dpi=300)
    plt.close(fig2)
    print('Plot saved to ' + plot2_path)
    print('\n--- Heatmap Summary Statistics ---')
    print('Analytic Phi(s): min = ' + str(round(phi_grid.min(), 4)) + ', max = ' + str(round(phi_grid.max(), 4)))
    print('Condition A Learned V(s): min = ' + str(round(q_grid_A.min(), 4)) + ', max = ' + str(round(q_grid_A.max(), 4)))
    print('Condition B Learned V(s): min = ' + str(round(q_grid_B.min(), 4)) + ', max = ' + str(round(q_grid_B.max(), 4)))
    print('Condition B Residual f_theta(s): min = ' + str(round(residual_grid_B.min(), 4)) + ', max = ' + str(round(residual_grid_B.max(), 4)))
    print('Mean Absolute Residual for Condition B: ' + str(round(np.mean(np.abs(residual_grid_B)), 4)))

if __name__ == '__main__':
    main()