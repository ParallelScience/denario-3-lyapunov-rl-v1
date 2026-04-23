1. **Environment and Reward Configuration**:
   - Utilize `gymnasium.make('Pendulum-v1')` with a custom reward wrapper replacing the native reward with:
     $$R_t = \Phi(s_t) - \Phi(s_{t+1}), \quad \Phi(s) = (1 - \cos\theta) + 0.5\dot\theta^2$$
   - Recover $\theta = \text{atan2}(\sin\theta, \cos\theta)$ from the state.

2. **PPO Implementation**:
   - Implement Proximal Policy Optimization (PPO) with a separate actor and critic.
   - The critic estimates the **state value function V(s)** — not Q(s,a). This is the correct architecture for the $V(s) = \Phi(s) + f_\theta(s)$ decomposition.
   - Standard PPO hyperparameters: clip ratio $\epsilon=0.2$, GAE ($\lambda=0.95$), $\gamma=0.99$, entropy coefficient $c_2=0.01$, 4 epochs per rollout, rollout length 2048, minibatch size 64.

3. **Three Experimental Conditions**:
   - **Condition A (Baseline)**: Standard PPO critic — 2-layer MLP outputs $V(s)$ directly.
   - **Condition B (Hard Structured Prior)**: Critic decomposed as $V(s) = \alpha \cdot \Phi(s) + f_\theta(s)$, where $\alpha$ is a **learnable scalar** (Softplus-constrained to be positive, initialized to 1.0). The residual network $f_\theta$ has its final layer initialized to zero. A separate optimizer with 10× lower learning rate trains $\alpha$ to prevent the prior from destabilizing early training.
   - **Condition C (Soft Regularization)**: Standard PPO critic (same as A), but the critic loss includes a soft Lyapunov regularization term: $\mathcal{L}_{total} = \mathcal{L}_{PPO} + \lambda \| V(s) - \Phi(s) \|^2$, with $\lambda=0.1$.

4. **Training Protocol**:
   - Train all three conditions for 100,000 environment steps across 5 random seeds each.
   - Use identical actor architecture and all other hyperparameters across conditions.
   - Log episode returns (Lyapunov reward), upright stability, and critic loss at each update.

5. **Absolute Performance Threshold (Sample Efficiency)**:
   - Replace the "90% of own max" metric with an **absolute return threshold** of −0.5.
   - Record steps to first consistently exceed this threshold (over a rolling 5-episode window).
   - This avoids penalizing conditions with higher performance ceilings.

6. **Gradient Monitoring (First 10,000 Steps)**:
   - For Condition B: log the norm of gradients for $f_\theta$ and the magnitude of the $\alpha \cdot \Phi(s)$ contribution at each update.
   - Log the value of $\alpha$ throughout training to track whether the prior is retained, amplified, or attenuated.

7. **Value Function Analysis**:
   - Evaluate $V(s)$ for all three conditions on a 100×100 grid of states ($\theta \in [-\pi, \pi]$, $\dot\theta \in [-8, 8]$).
   - Plot heatmaps of: (a) analytic $\Phi(s)$, (b) $V_A(s)$, (c) $V_B(s) = \alpha\Phi(s) + f_\theta(s)$, (d) residual $f_\theta(s)$, (e) learned $\alpha$ scalar over training.
   - Compute MSE between each learned $V(s)$ and $\Phi(s)$ on the grid.

8. **Statistical Comparison**:
   - Compare all three conditions on: upright stability (fraction of steps with $|\theta| < 0.1$ rad), steps to absolute threshold, final episode return, and critic convergence.
   - Report mean ± 95% CI across 5 seeds for each metric.
   - Assess whether adaptive $\alpha$ (Condition B) or soft regularization (Condition C) is more effective at resolving the bimodal failure mode observed in Iter 1.
