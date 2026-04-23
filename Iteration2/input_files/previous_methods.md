1. **Environment and Reward Configuration**:
   - Utilize `Pendulum-v1`. Implement a reward wrapper that replaces the native reward with the Lyapunov-based reward:
     $$R_t = \Phi(s_t) - \Phi(s_{t+1}), \quad \Phi(s) = (1 - \cos\theta) + 0.5\dot\theta^2$$
   - Recover $\theta$ using `atan2(sin θ, cos θ)` from the state representation.

2. **PPO Implementation**:
   - Implement Proximal Policy Optimization (PPO) with a shared or separate actor-critic architecture.
   - The critic explicitly estimates the state value function $V(s)$ — not $Q(s,a)$ — making it a natural fit for the structured decomposition below.
   - Use standard PPO hyperparameters: clip ratio $\epsilon=0.2$, entropy bonus, GAE ($\lambda=0.95$), $\gamma=0.99$.

3. **Condition A — Direct Value Learning (Baseline)**:
   - Standard PPO critic: a 2-layer MLP that outputs $V(s)$ directly.
   - Trained end-to-end on the Lyapunov reward.

4. **Condition B — Structured Value Function**:
   - The PPO critic is decomposed as $V(s) = \Phi(s) + f_\theta(s)$.
   - The network outputs only the residual $f_\theta(s)$; the analytic $\Phi(s)$ is added to the output.
   - **Initialization**: set the final layer weights and biases of $f_\theta$ to zero, so $V(s) \approx \Phi(s)$ at the start of training.
   - The policy (actor) is identical in both conditions; only the critic architecture differs.

5. **Training Protocol**:
   - Train both conditions for 100,000 environment steps across 5 random seeds each.
   - Use identical hyperparameters for both conditions.
   - Log episode returns (Lyapunov reward), upright stability, and critic loss at each update.

6. **Value Function Analysis**:
   - Evaluate the learned $V(s)$ for both conditions on a 2D grid of states ($\theta \in [-\pi, \pi]$, $\dot\theta \in [-8, 8]$).
   - Plot heatmaps of: (a) analytic $\Phi(s)$, (b) learned $V_A(s)$, (c) learned $V_B(s)$, (d) learned residual $f_\theta(s)$.
   - Compare the residual $f_\theta(s)$ to the theoretical expectation $V^*(s) - \Phi(s)$ to assess whether the network learns the "correction" beyond the Lyapunov prior.

7. **Performance Metrics**:
   - Learning curves: mean ± std Lyapunov reward over training steps (5 seeds).
   - Sample efficiency: steps to reach 90% of maximum average reward.
   - Upright stability: fraction of evaluation steps with $|\theta| < 0.1$ rad.
   - Critic convergence: MSE of $V(s)$ against a high-fidelity $V^*$ baseline (trained for 500k steps) over training time.

8. **Statistical Comparison**:
   - Compare Condition A vs. Condition B on all metrics using confidence intervals across 5 seeds.
   - Assess whether the Lyapunov structural prior accelerates critic convergence even when the policy fails to reach the upright position.
