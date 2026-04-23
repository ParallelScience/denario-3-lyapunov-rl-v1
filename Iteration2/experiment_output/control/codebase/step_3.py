# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import time
DATA_DIR = "data/"
TIMESTAMP = str(int(time.time()))
TOTAL_STEPS = 100000
SMOOTH_WINDOW = 10
GRID_SIZE = 100
def load_metrics(cond):
    path = os.path.join(DATA_DIR, "metrics_" + cond + ".npz")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
def load_grid():
    path = os.path.join(DATA_DIR, "grid_analysis.npz")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
def smooth_curve(values, window):
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode='same')
def interpolate_to_grid(ep_steps, ep_rets, common_steps):
    if len(ep_steps) == 0:
        return np.zeros_like(common_steps, dtype=np.float32)
    sm = smooth_curve(ep_rets.astype(np.float32), SMOOTH_WINDOW)
    return np.interp(common_steps, ep_steps.astype(np.float64), sm.astype(np.float64)).astype(np.float32)
def build_interp_matrix(metrics, common_steps):
    ep_steps_all = metrics['ep_steps']
    ep_rets_all = metrics['ep_returns']
    n_seeds = len(ep_steps_all)
    mat = np.zeros((n_seeds, len(common_steps)), dtype=np.float32)
    for i in range(n_seeds):
        mat[i] = interpolate_to_grid(ep_steps_all[i], ep_rets_all[i], common_steps)
    return mat
def build_critic_loss_matrix(metrics, common_upd):
    upd_steps_all = metrics['update_steps']
    c_losses_all = metrics['critic_losses']
    n_seeds = len(upd_steps_all)
    mat = np.zeros((n_seeds, len(common_upd)), dtype=np.float32)
    for i in range(n_seeds):
        us = upd_steps_all[i].astype(np.float64)
        cl = c_losses_all[i].astype(np.float64)
        if len(us) == 0:
            continue
        mat[i] = np.interp(common_upd, us, cl).astype(np.float32)
    return mat
def compute_mse(pred, target):
    return float(np.mean((pred - target) ** 2))
if __name__ == '__main__':
    metrics_A = load_metrics('A')
    metrics_B = load_metrics('B')
    grid = load_grid()
    phi_grid = grid['phi']
    VA_grid = grid['VA']
    VB_grid = grid['VB']
    fB_grid = grid['fB']
    mse_A = compute_mse(VA_grid, phi_grid)
    mse_B = compute_mse(VB_grid, phi_grid)
    fB_mean_abs = float(np.mean(np.abs(fB_grid)))
    phi_mean = float(np.mean(phi_grid))
    fB_rel = fB_mean_abs / (phi_mean + 1e-8)
    seff_A = metrics_A['sample_efficiency'].astype(np.float64)
    seff_B = metrics_B['sample_efficiency'].astype(np.float64)
    eval_rets_A = np.array([r for r in metrics_A['eval_returns']], dtype=object)
    eval_rets_B = np.array([r for r in metrics_B['eval_returns']], dtype=object)
    eval_ups_A = np.array([r for r in metrics_A['eval_upright_fracs']], dtype=object)
    eval_ups_B = np.array([r for r in metrics_B['eval_upright_fracs']], dtype=object)
    mean_eval_ret_A = np.array([np.mean(r) for r in eval_rets_A], dtype=np.float32)
    mean_eval_ret_B = np.array([np.mean(r) for r in eval_rets_B], dtype=np.float32)
    mean_eval_up_A = np.array([np.mean(r) for r in eval_ups_A], dtype=np.float32)
    mean_eval_up_B = np.array([np.mean(r) for r in eval_ups_B], dtype=np.float32)
    print("=== VALUE FUNCTION GRID ANALYSIS ===")
    print("MSE(V_A, Phi): " + str(round(mse_A, 6)))
    print("MSE(V_B, Phi): " + str(round(mse_B, 6)))
    print("Mean |f_theta(s)|: " + str(round(fB_mean_abs, 6)))
    print("Mean Phi(s): " + str(round(phi_mean, 6)))
    print("Relative residual |f_theta| / Phi: " + str(round(fB_rel, 6)))
    print("\n=== SAMPLE EFFICIENCY (steps to 90% max reward) ===")
    print("Condition A per seed: " + str(seff_A.tolist()))
    print("Condition A mean +/- std: " + str(round(float(np.mean(seff_A)), 1)) + " +/- " + str(round(float(np.std(seff_A)), 1)))
    print("Condition B per seed: " + str(seff_B.tolist()))
    print("Condition B mean +/- std: " + str(round(float(np.mean(seff_B)), 1)) + " +/- " + str(round(float(np.std(seff_B)), 1)))
    print("\n=== EVALUATION RETURNS (mean over 10 eval episodes) ===")
    print("Condition A per seed: " + str([round(float(v), 4) for v in mean_eval_ret_A]))
    print("Condition A mean +/- std: " + str(round(float(np.mean(mean_eval_ret_A)), 4)) + " +/- " + str(round(float(np.std(mean_eval_ret_A)), 4)))
    print("Condition B per seed: " + str([round(float(v), 4) for v in mean_eval_ret_B]))
    print("Condition B mean +/- std: " + str(round(float(np.mean(mean_eval_ret_B)), 4)) + " +/- " + str(round(float(np.std(mean_eval_ret_B)), 4)))
    print("\n=== UPRIGHT STABILITY (fraction of steps with |theta| < 0.1 rad) ===")
    print("Condition A per seed: " + str([round(float(v), 4) for v in mean_eval_up_A]))
    print("Condition A mean +/- std: " + str(round(float(np.mean(mean_eval_up_A)), 4)) + " +/- " + str(round(float(np.std(mean_eval_up_A)), 4)))
    print("Condition B per seed: " + str([round(float(v), 4) for v in mean_eval_up_B]))
    print("Condition B mean +/- std: " + str(round(float(np.mean(mean_eval_up_B)), 4)) + " +/- " + str(round(float(np.std(mean_eval_up_B)), 4)))
    common_steps = np.linspace(0, TOTAL_STEPS, 500, dtype=np.float64)
    mat_A = build_interp_matrix(metrics_A, common_steps)
    mat_B = build_interp_matrix(metrics_B, common_steps)
    mean_A = mat_A.mean(axis=0)
    std_A = mat_A.std(axis=0)
    mean_B = mat_B.mean(axis=0)
    std_B = mat_B.std(axis=0)
    color_A = '#1f77b4'
    color_B = '#d62728'
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    n_seeds = mat_A.shape[0]
    for i in range(n_seeds):
        ax1.plot(common_steps / 1000, mat_A[i], color=color_A, alpha=0.2, linewidth=0.8)
        ax1.plot(common_steps / 1000, mat_B[i], color=color_B, alpha=0.2, linewidth=0.8)
    ax1.fill_between(common_steps / 1000, mean_A - std_A, mean_A + std_A, color=color_A, alpha=0.25)
    ax1.fill_between(common_steps / 1000, mean_B - std_B, mean_B + std_B, color=color_B, alpha=0.25)
    ax1.plot(common_steps / 1000, mean_A, color=color_A, linewidth=2.0, label='Condition A (Direct)')
    ax1.plot(common_steps / 1000, mean_B, color=color_B, linewidth=2.0, label='Condition B (Structured)')
    ax1.set_xlabel('Environment Steps (x1000)', fontsize=12)
    ax1.set_ylabel('Episode Return (Lyapunov reward, dimensionless)', fontsize=12)
    ax1.set_title('Learning Curves: Condition A vs. B (5 seeds, mean +/- std)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.4)
    plt.tight_layout()
    path1 = os.path.join(DATA_DIR, "learning_curves_1_" + TIMESTAMP + ".png")
    fig1.savefig(path1, dpi=300)
    plt.close(fig1)
    print("\nLearning curves saved to " + path1)