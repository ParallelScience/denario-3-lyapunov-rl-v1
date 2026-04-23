1. **Environment and Reward Configuration**:
   - Utilize `gymnasium.make('Pendulum-v1')` with a custom reward wrapper replacing the native reward with:
     $$R_t = \Phi(s_t) - \Phi(s_{t+1}), \quad \Phi(s) = (1 - \cos\theta) + 0.5\dot\theta^2$$
   - Recover $\theta = \text{atan2}(\sin\theta, \cos\theta)$ from the state.

2. **PPO Implementation**:
   - Implement Proximal Policy Optimization (PPO) with a separate actor and critic.
   - The critic estimates the **state value function V(s)** directly — this is the key architectural difference from Iter 1, which incorrectly used SAC (a Q-function method).
   - Standard PPO hyperparameters: clip ratio $\epsilon=0.2$, GAE ($\lambda=0.95$), $\gamma=0.99$, entropy coefficient $c_2=0.01$, 4 epochs per rollout, rollout length 2048, minibatch size 64.

3. **Condition A — Direct Value Learning (Baseline)**:
   - Standard PPO critic: a 2-layer MLP that outputs $V(s)$ directly, learned entirely from data.

4. **Condition B — Structured Value Function**:
   - The PPO critic is decomposed as $V(s) = \Phi(s) + f_\theta(s)$.
   - The network outputs only the residual $f_\theta(s)$; the analytic $\Phi(s)$ is added to the output.
   - The residual network $f_\theta$ uses standard random initialization.
   - The actor is identical in both conditions; only the critic architecture differs.

5. **Training Protocol**:
   - Train both conditions for 100,000 environment steps across 5 random seeds each.
   - Use identical hyperparameters for both conditions.
   - Log episode returns (Lyapunov reward), upright stability, and critic loss at each update.

6. **Value Function Analysis**:
   - Evaluate the learned $V(s)$ for both conditions on a 100×100 grid of states ($\theta \in [-\pi, \pi]$, $\dot\theta \in [-8, 8]$).
   - Plot heatmaps of: (a) analytic $\Phi(s)$, (b) $V_A(s)$, (c) $V_B(s)$, (d) residual $f_\theta(s)$.
   - Compute MSE between each learned $V(s)$ and $\Phi(s)$ on the grid.

7. **Performance Metrics**:
   - Learning curves: mean ± std Lyapunov reward over training steps (5 seeds).
   - Sample efficiency: steps to reach 90% of maximum average reward.
   - Upright stability: fraction of evaluation steps with $|\theta| < 0.1$ rad.

8. **Statistical Comparison**:
   - Compare Condition A vs. Condition B on all metrics using mean ± std across 5 seeds.
   - Assess whether the Lyapunov structural prior accelerates critic convergence and improves policy quality.
