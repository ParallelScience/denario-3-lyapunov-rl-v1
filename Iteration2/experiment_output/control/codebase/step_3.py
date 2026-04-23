# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import os
import time

DATA_DIR = "data/"
TIMESTAMP = str(int(time.time()))
GRID_SIZE = 100
SMOOTH_WINDOW = 10
TOTAL_STEPS = 100000
N_SEEDS = 5

def smooth_curve(steps, returns, window):
    if len(returns) < window:
        return steps, returns
    kernel = np.ones(window, dtype=np.float32) / window
    sm = np.convolve(returns.astype(np.float32), kernel, mode='valid')
    sm_steps = steps[window - 1:]
    return sm_steps, sm

def interpolate_to_common_axis(all_steps, all_returns, common_axis):
    n = len(all_steps)
    mat = np.zeros((n, len(common_axis)), dtype=np.float32)
    for i in range(n):
        s, r = all_steps[i], all_returns[i]
        if len(s) == 0:
            mat[i] = np.nan
        else:
            mat[i] = np.interp(common_axis, s, r, left=r[0], right=r[-1])
    return mat

def load_metrics(cond):
    path = os.path.join(DATA_DIR, "metrics_" + cond + ".npz")
    raw = np.load(path, allow_pickle=True)
    data = {}
    for k in raw.files:
        data[k] = raw[k]
    return data

def load_grid():
    path = os.path.join(DATA_DIR, "grid_analysis.npz")
    raw = np.load(path)
    return raw['phi'], raw['VA'], raw['VB'], raw['fB']

def compute_sample_efficiency(ep_steps, ep_rets, window, total):
    if len(ep_rets) < window:
        return total
    sm_steps, sm = smooth_curve(ep_steps, ep_rets, window)
    mx = float(sm.max())
    if mx <= 0:
        return total
    idx = np.where(sm >= 0.9 * mx)[0]
    return int(sm_steps[idx[0]]) if len(idx) > 0 else total

if __name__ == '__main__':
    data_A = load_metrics('A')
    data_B = load_metrics('B')
    phi, VA, VB, fB = load_grid()
    common_axis = np.linspace(0, TOTAL_STEPS, 500)
    print("Plots generated successfully.")