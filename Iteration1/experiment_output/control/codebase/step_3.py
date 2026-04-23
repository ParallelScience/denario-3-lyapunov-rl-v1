# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import time

data_dir = 'data/'
SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = ['A', 'B']
CONDITION_LABELS = {'A': 'Cond. A (Direct)', 'B': 'Cond. B (Structured)'}
CONDITION_COLORS = {'A': '#1f77b4', 'B': '#d62728'}
SMOOTH_WINDOW = 20

def smooth_series(arr, window):
    kernel = np.ones(window) / window
    padded = np.concatenate([arr[:window - 1], arr])
    smoothed_full = np.convolve(padded, kernel, mode='valid')
    return smoothed_full[:len(arr)]

def interpolate_to_common_steps(steps_list, values_list, n_points=500):
    global_min = max(s[0] for s in steps_list)
    global_max = min(s[-1] for s in steps_list)
    common_steps = np.linspace(global_min, global_max, n_points)
    interp_matrix = np.zeros((len(steps_list), n_points))
    for i, (steps, vals) in enumerate(zip(steps_list, values_list)):
        interp_matrix[i] = np.interp(common_steps, steps, vals)
    return common_steps, interp_matrix

def load_episode_logs(conditions, seeds, data_dir):
    logs = {}
    for cond in conditions:
        logs[cond] = {}
        for seed in seeds:
            path = os.path.join(data_dir, 'log_' + cond + '_' + str(seed) + '.csv')
            logs[cond][seed] = pd.read_csv(path)
    return logs

def compute_curves(logs, cond, seeds, key, smooth_window, n_points=500):
    steps_list, raw_list, smooth_list = [], [], []
    for seed in seeds:
        df = logs[cond][seed]
        steps = df['step'].values.astype(float)
        vals = df[key].values.astype(float)
        smoothed = smooth_series(vals, smooth_window)
        steps_list.append(steps)
        raw_list.append(vals)
        smooth_list.append(smoothed)
    common_steps, raw_matrix = interpolate_to_common_steps(steps_list, raw_list, n_points)
    _, smooth_matrix = interpolate_to_common_steps(steps_list, smooth_list, n_points)
    return (common_steps, np.mean(smooth_matrix, axis=0), np.std(smooth_matrix, axis=0), np.mean(raw_matrix, axis=0), np.std(raw_matrix, axis=0))

if __name__ == '__main__':
    timestamp = str(int(time.time()))
    logs = load_episode_logs(CONDITIONS, SEEDS, data_dir)
    fig, ax = plt.subplots(figsize=(11, 5))
    curve_data = {}
    for cond in CONDITIONS:
        curve_data[cond] = compute_curves(logs, cond, SEEDS, 'episode_return', SMOOTH_WINDOW)
    global_max = max(np.max(curve_data[c][1]) for c in CONDITIONS)
    ref_line = 0.9 * global_max
    for cond in CONDITIONS:
        common_steps, smooth_mean, smooth_std, raw_mean, raw_std = curve_data[cond]
        color = CONDITION_COLORS[cond]
        label = CONDITION_LABELS[cond]
        ax.fill_between(common_steps, raw_mean - raw_std, raw_mean + raw_std, alpha=0.18, color=color, label=label + ' raw std band')
        ax.plot(common_steps, smooth_mean, color=color, linewidth=2.2, label=label + ' smoothed')
    ax.axhline(ref_line, color='black', linestyle='--', linewidth=1.3, label='90% of max avg reward')
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title('Learning Curves')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(data_dir, 'learning_curves_' + timestamp + '.png'), dpi=300)
    plt.close(fig)
    print('Saved plots to data/')