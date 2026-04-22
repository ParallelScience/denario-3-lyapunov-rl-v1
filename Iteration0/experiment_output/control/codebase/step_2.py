# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
import numpy as np
import torch
import gymnasium as gym
import multiprocessing as mp
import pickle
from step_1 import SAC, SACArgs, ReplayMemory, LyapunovRewardWrapper
def train_agent(args_tuple):
    torch.set_num_threads(1)
    seed, condition = args_tuple
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    if condition == 'condition_A':
        env = LyapunovRewardWrapper(env)
        eval_env = LyapunovRewardWrapper(eval_env)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args = SACArgs()
    args.gamma = 0.99
    args.tau = 0.005
    args.alpha = 0.2
    args.target_update_interval = 1
    args.hidden_size = 256
    args.lr = 3e-4
    args.automatic_entropy_tuning = True
    args.structured = False
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    memory = ReplayMemory(100000, env.observation_space.shape[0], env.action_space.shape[0])
    total_numsteps = 0
    updates = 0
    num_steps = 100000
    batch_size = 256
    logs = {'episode_rewards': [], 'episode_lengths': [], 'eval_steps': [], 'eval_upright_fractions': []}
    episode_reward = 0
    episode_steps = 0
    obs, _ = env.reset(seed=seed)
    while total_numsteps < num_steps:
        if total_numsteps < 1000:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        memory.push(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if len(memory) > 1000:
            agent.update_parameters(memory, batch_size, updates)
            updates += 1
        if terminated or truncated:
            logs['episode_rewards'].append(episode_reward)
            logs['episode_lengths'].append(episode_steps)
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
        if total_numsteps % 10000 == 0:
            eval_obs, _ = eval_env.reset(seed=seed + 100 + total_numsteps)
            upright_steps = 0
            for _ in range(1000):
                eval_action = agent.select_action(eval_obs, evaluate=True)
                eval_obs, eval_reward, eval_terminated, eval_truncated, eval_info = eval_env.step(eval_action)
                theta = np.arctan2(eval_obs[1], eval_obs[0])
                if abs(theta) < 0.1:
                    upright_steps += 1
                if eval_terminated or eval_truncated:
                    eval_obs, _ = eval_env.reset()
            upright_frac = upright_steps / 1000.0
            logs['eval_steps'].append(total_numsteps)
            logs['eval_upright_fractions'].append(upright_frac)
        if total_numsteps % 20000 == 0:
            print('Task ' + condition + ' seed ' + str(seed) + ': ' + str(total_numsteps) + '/' + str(num_steps) + ' steps')
    torch.save(agent.critic.state_dict(), os.path.join('data', condition + '_critic_seed_' + str(seed) + '.pth'))
    torch.save(agent.policy.state_dict(), os.path.join('data', condition + '_policy_seed_' + str(seed) + '.pth'))
    avg_last_10_rewards = np.mean(logs['episode_rewards'][-10:]) if len(logs['episode_rewards']) >= 10 else np.mean(logs['episode_rewards'])
    print('Completed Condition: ' + condition + ', Seed: ' + str(seed) + ' | Final Avg Reward (last 10 eps): ' + str(round(avg_last_10_rewards, 2)) + ' | Final Upright Fraction: ' + str(logs['eval_upright_fractions'][-1]))
    return condition, seed, logs
if __name__ == '__main__':
    tasks = []
    for seed in range(5):
        tasks.append((seed, 'vanilla'))
        tasks.append((seed, 'condition_A'))
    print('Starting training for Vanilla SAC and Condition A (10 tasks total, parallel on CPU)...')
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=10) as pool:
        results = pool.map(train_agent, tasks)
    print('Training completed. Saving logs...')
    all_logs = {'vanilla': {}, 'condition_A': {}}
    for condition, seed, logs in results:
        all_logs[condition][seed] = logs
    with open(os.path.join('data', 'step_2_logs.pkl'), 'wb') as f:
        pickle.dump(all_logs, f)
    print('Logs saved to data/step_2_logs.pkl')
    for condition in ['vanilla', 'condition_A']:
        final_rewards = [np.mean(all_logs[condition][seed]['episode_rewards'][-10:]) for seed in range(5)]
        final_uprights = [all_logs[condition][seed]['eval_upright_fractions'][-1] for seed in range(5)]
        print('\nSummary for ' + condition + ':')
        print('Mean Final Avg Reward (last 10 eps): ' + str(round(np.mean(final_rewards), 2)) + ' +/- ' + str(round(np.std(final_rewards), 2)))
        print('Mean Final Upright Fraction: ' + str(round(np.mean(final_uprights), 4)) + ' +/- ' + str(round(np.std(final_uprights), 4)))