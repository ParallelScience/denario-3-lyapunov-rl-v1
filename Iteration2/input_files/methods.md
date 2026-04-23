1. **Environment and SAC Architecture**:
   - Utilize `gymnasium.make('Pendulum-v1')` with a custom reward wrapper implementing $R_t = \Phi(s_t) - \Phi(s_{t+1})$.
   - Implement SAC using a standard PyTorch framework, ensuring double-Q learning (two Q-networks) is applied consistently across all conditions.
   - For Condition A (Baseline), use standard MLP critics $Q_\theta(s, a)$.
   - For Condition B (Structured), implement the decomposition $Q(s, a) = \alpha \Phi(s) + f_\theta(s, a)$, where $\alpha$ is a learnable scalar (initialized to 0.1 to prevent initial bias) and $f_\theta(s, a)$ is the residual MLP. Apply this to both online and target Q-networks.

2. **Initialization and Learning Rate Scheduling**:
   - Initialize the final layer weights and biases of the residual network $f_\theta$ to zero so that $Q(s, a) \approx \alpha \Phi(s)$ at the start of training.
   - Use a separate optimizer for the learnable scalar $\alpha$ with a learning rate 10x smaller than the learning rate for the residual network $f_\theta$ to prevent the prior from destabilizing the critic during early learning.
   - Constrain $\alpha$ to be positive using a softplus activation.

3. **Soft Regularization (Condition C)**:
   - Implement a third condition where the Lyapunov prior is applied as a soft constraint: $L_{total} = L_{Bellman} + \lambda \cdot \text{MSE}(V_{critic}(s), \Phi(s))$.
   - Define $V_{critic}(s) = \mathbb{E}_{a \sim \pi}[Q(s, a)]$ using the current policy to ensure the regularization term is state-dependent and matches the scale of $\Phi(s)$.

4. **Training Protocol**:
   - Train all conditions (A, B, and C) for 100,000 environment steps.
   - Use 5 random seeds per condition.
   - Maintain identical hyperparameters (learning rate, batch size, buffer size, entropy coefficient $\alpha_{entropy}$) across all conditions to ensure a fair comparison.

5. **Performance Metrics and Thresholds**:
   - Calculate the maximum possible return per episode as $\Phi(s_{initial})$ based on the Pendulum's starting state distribution.
   - Set the absolute performance threshold for sample efficiency at 80% of the average $\Phi(s_{initial})$.
   - Record the number of environment steps required to consistently cross this threshold.

6. **Gradient and Sensitivity Analysis**:
   - Monitor the evolution of the learnable scalar $\alpha$ in Condition B throughout training. Log its value at every gradient update to determine if the model tunes the prior or if it collapses to zero.
   - During the first 10,000 steps, log the norm of the gradients for $f_\theta$ and compare them against the gradients of the fixed $\Phi(s)$ component to assess the learning dynamics.

7. **Value Function and Policy Evaluation**:
   - Evaluate the learned $Q(s, a)$ on a 2D grid of states $(\theta, \dot\theta)$ by fixing $a=0$.
   - Calculate upright stability as the fraction of evaluation steps where $|\theta| < 0.1$ rad.
   - Compare final policy quality by rendering the agent and calculating the average return over 100 evaluation episodes.

8. **Statistical Comparison**:
   - Perform a comparative analysis of the three conditions using mean and 95% confidence intervals across the 5 seeds.
   - Use the collected metrics to determine if the learnable scalar (Condition B) or soft regularization (Condition C) provides superior stability and sample efficiency compared to the baseline.