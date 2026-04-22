1. **Environment Setup and Reward Wrapper**:
   - Instantiate the `Pendulum-v1` environment.
   - Implement a wrapper to replace the native reward with the Lyapunov-based reward $R_t = \Phi(s_t) - \Phi(s_{t+1})$, where $\Phi(s) = (1 - \cos\theta) + 0.5\dot\theta^2$.
   - Ensure the state representation correctly computes $\theta = \text{atan2}(\sin\theta, \cos\theta)$ to avoid discontinuities.

2. **Baseline Implementation (Condition A)**:
   - Configure a standard SAC agent with an MLP critic network predicting $V(s)$.
   - Enable automated entropy tuning (`alpha='auto'`) to ensure the agent adapts to the scale of the Lyapunov reward.

3. **Structured Critic Implementation (Condition B)**:
   - Modify the SAC critic architecture to implement $V(s) = \Phi(s) + f_\theta(s)$.
   - The critic network will output only the residual $f_\theta(s)$.
   - **Initialization**: Initialize the weights of the final layer of the residual network $f_\theta(s)$ to zero (or near-zero) to ensure $V(s) \approx \Phi(s)$ at the start of training.

4. **Training Protocol**:
   - Execute training for both conditions for 100,000 environment steps across 5 random seeds.
   - Use identical hyperparameters (learning rate, batch size, $\gamma$) for both conditions.
   - Include a "Vanilla SAC" control group (using the original Pendulum-v1 reward) to provide a reference point for the difficulty of the task.

5. **Data Collection and Periodic Evaluation**:
   - Log cumulative Lyapunov reward and episode length.
   - Perform periodic evaluation every 10,000 steps: calculate the fraction of time steps where the pendulum is in the upright region ($|\theta| < 0.1$ rad) over a 1,000-step evaluation roll-out.
   - Save model checkpoints at these intervals.

6. **Value Function and Residual Analysis**:
   - Generate a 2D grid of states ($\theta \in [-\pi, \pi], \dot\theta \in [-8, 8]$).
   - Compute and plot: (a) the analytic $\Phi(s)$, (b) the learned $V(s)$ for Condition A, (c) the learned $V(s)$ for Condition B, and (d) the learned residual $f_\theta(s)$ for Condition B.
   - Analyze the residual heatmap to identify what the network is "correcting" (e.g., friction or control effort).

7. **Performance Metrics**:
   - Calculate mean and standard deviation of learning curves across 5 seeds.
   - Determine sample efficiency: steps required to reach 90% of the maximum average reward observed across all conditions.
   - Quantify final upright stability as the percentage of time spent in the goal region during the final 10,000 steps of training.

8. **Statistical Comparison**:
   - Perform a comparative analysis between Condition A and B.
   - Use the standard deviation across seeds to provide confidence intervals for all metrics.
   - Assess whether the structured approach (Condition B) provides a statistically significant improvement in convergence speed and final policy performance compared to the baseline (Condition A).