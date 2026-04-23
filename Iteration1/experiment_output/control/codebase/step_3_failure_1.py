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
from datetime import datetime

data_dir = 'data/'

def load_training_logs(condition, n_seeds=5):
    logs = []
    for seed in range(n_seeds):
        path = os.path.join(data_dir, 'log_' + condition + '_' + str(seed) + '.csv')
        df = pd.read_csv(path)
        logs.append(df)
    return logs

def interpolate_to_common_grid(logs, key, n_points=500):
    all_min = max(df['step'].min() for df in logs)
    all_max = min(df['step'].max() for df in logs)
    common_steps = np.linspace(all_min, all_max, n_points)
    matrix = np.zeros((len(logs), n_points))
    for i, df in enumerate(logs):
        matrix[i] = np.interp(common_steps, df['step'].values, df[key].values)
    return common_steps, matrix

def rolling_mean(arr, window=20):
    result = np.convolve(arr, np.ones(window) / window, mode='same')
    half = window // 2
    for i in range(half):
        result[i] = arr[:i + 1].mean()
        result[-(i + 1)] = arr[-(i + 1):].mean()
    return result

def plot_summary_metrics(save_path):
    df_eval = pd.read_csv(os.path.join(data_dir, 'eval_results.csv'))
    df_eff = pd.read_csv(os.path.join(data_dir, 'sample_efficiency.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, (metric, title, ax) in enumerate([('upright_fraction', 'Upright Stability', axes[0]), ('steps_to_90', 'Sample Efficiency', axes[1])]):
        means = [df_eval[df_eval['condition'] == c][metric].mean() for c in ['A', 'B']]
        stds = [df_eval[df_eval['condition'] == c][metric].std() for c in ['A', 'B']]
        ax.bar(['A', 'B'], means, yerr=stds, color=['steelblue', 'darkorange'], alpha=0.7, capsize=5)
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    logs_A = load_training_logs('A')
    logs_B = load_training_logs('B')
    plot_summary_metrics(os.path.join(data_dir, 'summary_metrics.png'))
    print('Saved to ' + os.path.join(data_dir, 'summary_metrics.png'))