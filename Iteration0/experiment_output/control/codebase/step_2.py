# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import torch
import gymnasium as gym
import pickle
import time
from step_1 import SAC, SACArgs, ReplayMemory, LyapunovRewardWrapper

def evaluate_policy(agent, env_name, wrapper, seed, eval_steps=1000):
    env = gym.make(env_name)
    if wrapper:
        env = LyapunovRewardWrapper(env)
    obs, _ = env.reset(seed=seed)
    upright_steps = 0
    for _ in range(eval_steps):
        action = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        theta = np.arctan2(obs[1], obs[0])
        if abs(theta) < 0.1:
            upright_steps += 1
        if terminated or truncated:
            obs, _ = env.reset()
    return upright_steps / eval_steps

def train_agent(seed, condition):
    print('Starting ' + condition + ' seed ' + str(seed), flush=True)
    start_time = time.time()
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    if condition == 'condition_A':
        env = LyapunovRewardWrapper(env)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = SACArgs()
    args.gamma = 0.99
    args.tau = 0.005
    args.lr = 3e-4
    args.hidden_size = 256
    args.automatic_entropy_tuning = True
    args.structured = False
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    memory = ReplayMemory(100000, env.observation_space.shape[0], env.action_space.shape[0])
    updates = 0
    num_steps = 30000
    batch_size = 256
    start_steps = 1000
    update_frequency = 1
    eval_frequency = 5000
    episode_rewards = []
    episode_lengths = []
    episode_end_steps = []
    eval_metrics = []
    eval_steps_log = []
    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_length = 0
    for total_numsteps in range(1, num_steps + 1):
        if total_numsteps <= start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_length += 1
        if condition == 'vanilla':
            theta_t = np.arctan2(obs[1], obs[0])
            phi_t = (1.0 - np.cos(theta_t)) + 0.5 * (obs[2] ** 2)
            theta_tp1 = np.arctan2(next_obs[1], next_obs[0])
            phi_tp1 = (1.0 - np.cos(theta_tp1)) + 0.5 * (next_obs[2] ** 2)
            lyapunov_reward = phi_t - phi_tp1
            episode_reward += lyapunov_reward
        else:
            episode_reward += reward
        memory.push(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if len(memory) > batch_size and total_numsteps % update_frequency == 0:
            agent.update_parameters(memory, batch_size, updates)
            updates += 1
        if terminated or truncated:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_end_steps.append(total_numsteps)
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        if total_numsteps % eval_frequency == 0:
            wrapper = (condition == 'condition_A')
            upright_frac = evaluate_policy(agent, env_name, wrapper, seed + 100)
            eval_metrics.append(upright_frac)
            eval_steps_log.append(total_numsteps)
            print('  Step ' + str(total_numsteps) + ' Upright Frac: ' + str(upright_frac), flush=True)
    model_path = os.path.join('data', 'model_' + condition + '_seed_' + str(seed) + '.pt')
    torch.save({'critic': agent.critic.state_dict(), 'policy': agent.policy.state_dict()}, model_path)
    res = {'condition': condition, 'seed': seed, 'episode_rewards': episode_rewards, 'episode_lengths': episode_lengths, 'episode_end_steps': episode_end_steps, 'eval_metrics': eval_metrics, 'eval_steps': eval_steps_log}
    print('Finished ' + condition + ' seed ' + str(seed) + ' in ' + str(round(time.time() - start_time, 2)) + 's', flush=True)
    del agent
    del memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res

if __name__ == '__main__':
    print('Starting sequential training for Vanilla SAC and Condition A...', flush=True)
    results = []
    for seed in range(5):
        for condition in ['vanilla', 'condition_A']:
            res = train_agent(seed, condition)
            results.append(res)
    with open(os.path.join('data', 'step_2_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print('\nTraining completed. Results saved to data/step_2_results.pkl', flush=True)
    print('\n--- Training Summary ---', flush=True)
    for res in results:
        cond = res['condition']
        seed = res['seed']
        final_eval = res['eval_metrics'][-1] if res['eval_metrics'] else 0
        avg_reward_last_10 = np.mean(res['episode_rewards'][-10:]) if len(res['episode_rewards']) >= 10 else np.mean(res['episode_rewards'])
        print('Condition: ' + cond + ', Seed: ' + str(seed) + ', Final Upright Frac: ' + str(final_eval) + ', Avg Reward (last 10 eps): ' + str(round(avg_reward_last_10, 2)), flush=True)