# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from step_1 import (lyapunov_phi, LyapunovPendulumWrapper, ActorNetwork, CriticNetworkA, CriticNetworkB)
DATA_DIR = 'data/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GAMMA, LAM, CLIP_EPS, ENTROPY_COEF, VALUE_COEF = 0.99, 0.95, 0.2, 0.01, 0.5
LR, ROLLOUT_LEN, MINIBATCH_SIZE, N_EPOCHS = 3e-4, 2048, 64, 4
TOTAL_STEPS, N_SEEDS, ACTION_SCALE = 100000, 5, 2.0
EVAL_EPISODES, GRID_SIZE, SMOOTH_WINDOW = 10, 100, 10
def make_env(seed):
    env = gym.make('Pendulum-v1')
    env = LyapunovPendulumWrapper(env)
    env.reset(seed=seed)
    return env
def gae(rewards, values, dones, nv, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        next_v = nv if t == T - 1 else float(values[t + 1])
        m = 1.0 - float(dones[t])
        d = float(rewards[t]) + gamma * next_v * m - float(values[t])
        g = d + gamma * lam * m * g
        adv[t] = np.float32(g)
    return adv, adv + values
def ppo_step(actor, critic, ao, co, st, at, lpt, advt, rett, cond, ne, mb):
    T = st.shape[0]
    ls = []
    actor.train(); critic.train()
    for _ in range(ne):
        p = torch.randperm(T, device=DEVICE)
        for s in range(0, T, mb):
            i = p[s:s + mb]
            ss, aa, lo, av, re = st[i], at[i], lpt[i], advt[i], rett[i]
            d = actor.get_dist(ss)
            r = torch.exp(d.log_prob(aa).sum(-1) - lo)
            al = -torch.min(r * av, torch.clamp(r, 1 - CLIP_EPS, 1 + CLIP_EPS) * av).mean()
            al = al - ENTROPY_COEF * d.entropy().sum(-1).mean()
            ao.zero_grad(); al.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 0.5); ao.step()
            vp = critic(ss) if cond == 'A' else critic.get_value(ss)
            cl = VALUE_COEF * nn.functional.mse_loss(vp, re)
            co.zero_grad(); cl.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 0.5); co.step()
            ls.append(cl.item())
    return float(np.mean(ls))
def train_seed(cond, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    env = make_env(seed)
    actor = ActorNetwork(3, 1).to(DEVICE)
    critic = (CriticNetworkA(3) if cond == 'A' else CriticNetworkB(3)).to(DEVICE)
    ao, co = optim.Adam(actor.parameters(), lr=LR), optim.Adam(critic.parameters(), lr=LR)
    esl, erl, eul, usl, cll = [], [], [], [], []
    done_steps, obs, er, eu, el = 0, env.reset()[0], 0.0, 0, 0
    while done_steps < TOTAL_STEPS:
        rl = min(ROLLOUT_LEN, TOTAL_STEPS - done_steps)
        sb, ab, lb, rb, vb, db = np.zeros((rl, 3), dtype=np.float32), np.zeros((rl, 1), dtype=np.float32), np.zeros(rl, dtype=np.float32), np.zeros(rl, dtype=np.float32), np.zeros(rl, dtype=np.float32), np.zeros(rl, dtype=np.float32)
        actor.eval(); critic.eval()
        with torch.no_grad():
            for t in range(rl):
                o = obs.astype(np.float32)
                ot = torch.tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                dist = actor.get_dist(ot)
                ra = dist.sample()
                lp = dist.log_prob(ra).sum(-1).item()
                an = torch.clamp(ra, -ACTION_SCALE, ACTION_SCALE).cpu().numpy().flatten().astype(np.float32)
                v = (critic(ot) if cond == 'A' else critic.get_value(ot)).item()
                no, rw, te, tr, _ = env.step(an)
                dn = float(te or tr)
                th = float(np.arctan2(o[1], o[0]))
                eu += int(abs(th) < 0.1); er += float(rw); el += 1
                sb[t], ab[t], lb[t], rb[t], vb[t], db[t] = o, an, np.float32(lp), np.float32(rw), np.float32(v), np.float32(dn)
                obs = no
                if dn:
                    esl.append(done_steps + t + 1); erl.append(er); eul.append(eu / el if el > 0 else 0.0)
                    er, eu, el = 0.0, 0, 0; obs, _ = env.reset()
            o = obs.astype(np.float32)
            ot = torch.tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            nv = (critic(ot) if cond == 'A' else critic.get_value(ot)).item()
        adv, ret = gae(rb, vb, db, float(nv), GAMMA, LAM)
        at_ = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
        at_ = (at_ - at_.mean()) / (at_.std() + 1e-8)
        cl = ppo_step(actor, critic, ao, co, torch.tensor(sb, dtype=torch.float32, device=DEVICE), torch.tensor(ab, dtype=torch.float32, device=DEVICE), torch.tensor(lb, dtype=torch.float32, device=DEVICE), at_, torch.tensor(ret, dtype=torch.float32, device=DEVICE), cond, N_EPOCHS, MINIBATCH_SIZE)
        done_steps += rl; usl.append(done_steps); cll.append(cl)
    env.close()
    return np.array(esl, dtype=np.int64), np.array(erl, dtype=np.float32), np.array(eul, dtype=np.float32), np.array(usl, dtype=np.int64), np.array(cll, dtype=np.float32), actor, critic
def eval_policy(actor, seed):
    env = make_env(seed + 1000)
    actor.eval()
    rets, ups = [], []
    with torch.no_grad():
        for _ in range(EVAL_EPISODES):
            obs, _ = env.reset()
            er, eu, el, dn = 0.0, 0, 0, False
            while not dn:
                o = obs.astype(np.float32)
                ot = torch.tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                an = torch.clamp(actor.get_dist(ot).mean, -ACTION_SCALE, ACTION_SCALE).cpu().numpy().flatten().astype(np.float32)
                obs, rw, te, tr, _ = env.step(an)
                dn = te or tr
                th = float(np.arctan2(obs.astype(np.float32)[1], obs.astype(np.float32)[0]))
                eu += int(abs(th) < 0.1); er += float(rw); el += 1
            rets.append(er); ups.append(eu / el if el > 0 else 0.0)
    env.close()
    return np.array(rets, dtype=np.float32), np.array(ups, dtype=np.float32)
def sample_eff(ep_steps, ep_rets, sw, total):
    n = len(ep_rets)
    if n < sw: return total
    sm = np.convolve(ep_rets, np.ones(sw, dtype=np.float32) / sw, mode='valid')
    ss = ep_steps[sw - 1:]
    mx = float(sm.max())
    if mx <= 0: return total
    idx = np.where(sm >= 0.9 * mx)[0]
    return int(ss[idx[0]]) if len(idx) > 0 else total
if __name__ == '__main__':
    all_ep_steps, all_ep_rets, all_ep_ups, all_upd_steps, all_c_losses, all_eval_rets, all_eval_ups, all_seff = {}, {}, {}, {}, {}, {}, {}, {}
    seed0_actors, seed0_critics = {}, {}
    for cond in ['A', 'B']:
        all_ep_steps[cond], all_ep_rets[cond], all_ep_ups[cond], all_upd_steps[cond], all_c_losses[cond], all_eval_rets[cond], all_eval_ups[cond], all_seff[cond] = [], [], [], [], [], [], [], []
        for seed in range(N_SEEDS):
            es, er, eu, us, cl, actor, critic = train_seed(cond, seed)
            torch.save(actor.state_dict(), os.path.join(DATA_DIR, 'actor_' + cond + '_' + str(seed) + '.pth'))
            torch.save(critic.state_dict(), os.path.join(DATA_DIR, 'critic_' + cond + '_' + str(seed) + '.pth'))
            all_ep_steps[cond].append(es); all_ep_rets[cond].append(er); all_ep_ups[cond].append(eu); all_upd_steps[cond].append(us); all_c_losses[cond].append(cl)
            ev_r, ev_u = eval_policy(actor, seed)
            all_eval_rets[cond].append(ev_r); all_eval_ups[cond].append(ev_u)
            all_seff[cond].append(sample_eff(es, er, SMOOTH_WINDOW, TOTAL_STEPS))
            if seed == 0: seed0_actors[cond] = actor; seed0_critics[cond] = critic
    for cond in ['A', 'B']:
        np.savez(os.path.join(DATA_DIR, 'metrics_' + cond + '.npz'), ep_steps=np.array(all_ep_steps[cond], dtype=object), ep_returns=np.array(all_ep_rets[cond], dtype=object), ep_upright_fracs=np.array(all_ep_ups[cond], dtype=object), update_steps=np.array(all_upd_steps[cond], dtype=object), critic_losses=np.array(all_c_losses[cond], dtype=object), eval_returns=np.array(all_eval_rets[cond], dtype=object), eval_upright_fracs=np.array(all_eval_ups[cond], dtype=object), sample_efficiency=np.array(all_seff[cond], dtype=np.int64))
    th_v = np.linspace(-np.pi, np.pi, GRID_SIZE, dtype=np.float32)
    td_v = np.linspace(-8.0, 8.0, GRID_SIZE, dtype=np.float32)
    TH, TD = np.meshgrid(th_v, td_v, indexing='ij')
    st_f = np.stack([np.cos(TH).ravel(), np.sin(TH).ravel(), TD.ravel()], axis=1).astype(np.float32)
    st_t = torch.tensor(st_f, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        VA_f = seed0_critics['A'](st_t).cpu().numpy().astype(np.float32)
        fB_f = seed0_critics['B'](st_t).cpu().numpy().astype(np.float32)
        phi_f = lyapunov_phi(st_t).cpu().numpy().astype(np.float32)
        VB_f = (phi_f + fB_f).astype(np.float32)
    np.savez(os.path.join(DATA_DIR, 'grid_analysis.npz'), phi=phi_f.reshape(GRID_SIZE, GRID_SIZE), VA=VA_f.reshape(GRID_SIZE, GRID_SIZE), VB=VB_f.reshape(GRID_SIZE, GRID_SIZE), fB=fB_f.reshape(GRID_SIZE, GRID_SIZE))
    print('Training and evaluation complete.')