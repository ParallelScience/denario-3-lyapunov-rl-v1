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

data_dir = 'data/'

def load_training_logs(condition, n_seeds=5):
    logs = []
    for seed in range(n_seeds):
        path = os.path.join(data_dir, 'log_' + condition + '_' + str(seed) + '.csv')
        df = pd.read_csv(path)
        logs.append(df)
    return logs

def plot_summary_metrics(save_path):
    df_eval = pd.read_csv(os.path.join(data_dir, 'eval_results.csv'))
    df_eff = pd.read_csv(os.path.join(data_dir, 'sample_efficiency.csv'))
    print('eval_results.csv:\n', df_eval)
    print('sample_efficiency.csv:\n', df_eff)
    upright_col = [c for c in df_eval.columns if 'upright' in c.lower()][0]
    eff_col = [c for c in df_eff.columns if 'step' in c.lower()][0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, (df, col, title, ylabel, ax) in enumerate([(df_eval, upright_col, 'Upright Stability', 'Fraction', axes[0]), (df_eff, eff_col, 'Sample Efficiency', 'Steps', axes[1])]):
        means = [df[df['condition'] == c][col].mean() for c in ['A', 'B']]
        stds = [df[df['condition'] == c][col].std() for c in ['A', 'B']]
        ax.bar(['Direct (A)', 'Structured (B)'], means, yerr=[np.minimum(stds, means), stds], color=['steelblue', 'darkorange'], alpha=0.7, capsize=5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if i == 0:
            ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    plot_summary_metrics(os.path.join(data_dir, 'summary_metrics.png'))
    print('Saved to ' + os.path.join(data_dir, 'summary_metrics.png'))