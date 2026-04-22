1. **Environment and Reward Configuration**:
   - Utilize `Pendulum-v1`. Implement a reward wrapper that defines the total reward as the sum of the native environment reward and the potential-based shaping term: $R_{total} = R_{native} + (\gamma \Phi(s_{t+1}) - \Phi(s_t))$, where $\Phi(s) = (1 - \cos\theta) + 0.5\dot\theta^2$.
   - Ensure $\theta$ is recovered using `atan2(sin θ, cos θ)` to maintain consistency across all calculations.

2. **Baseline and Structured Critic Architectures**:
   - **Condition A (Baseline)**: Standard SAC critic $V_\psi(s)$ (MLP).
   - **Condition B (Structured)**: Critic defined as $V(s) = \Phi_\alpha(s) + f_\theta(s)$, where $\Phi_\alpha(s) = \alpha(1 - \cos\theta) + 0.5\dot\theta^2$.
   - **Initialization**: For $f_\theta(s)$, initialize the final layer weights and biases to zero. This ensures $V(s) = \Phi_\alpha(s)$ at the start of training.

3. **Robustness Testing (Model Mismatch)**:
   - Conduct experiments with $\alpha \in \{0.5, 1.0, 2.0\}$ for Condition B.
   - Keep the reward shaping term in Step 1 fixed to the "true" physics ($\alpha=1.0$) across all experiments to isolate the effect of the structural prior in the critic from the shaping signal.

4. **Ground Truth Value Function ($V^*$)**:
   - Generate a high-fidelity $V^*(s)$ by training a standard SAC agent for 500,000 steps. To ensure robustness, average the value function estimates across 5 independent seeds.
   - This $V^*(s)$ serves as the ground truth for calculating the "Residual Gap" for both Condition A ($V(s) - \Phi(s)$) and Condition B ($f_\theta(s)$).

5. **Training Protocol**:
   - Train all agents for 100,000 environment steps using 5 random seeds per configuration.
   - Maintain consistent hyperparameters (learning rate, batch size, $\gamma$) across all conditions. Monitor gradient magnitudes of $f_\theta$ early in training to ensure the network is actively learning the residual.

6. **Quantitative Residual Analysis**:
   - Define a fixed grid of the state space ($\theta \in [-\pi, \pi], \dot\theta \in [-8, 8]$).
   - At periodic intervals, compute the Mean Squared Error (MSE) between the learned residual and the target residual ($V^*(s) - \Phi(s)$).
   - Track this MSE over training time to quantify how effectively the neural network captures dynamics not represented by the prior.

7. **Performance Evaluation**:
   - **Convergence Rate**: Measure the MSE of the learned value function $V(s)$ relative to $V^*(s)$ over training steps.
   - **Stabilization Error**: Calculate the mean squared angular distance from the upright equilibrium ($\theta=0, \dot\theta=0$) during the final 10,000 steps of training, using the native state representation.
   - **Upright Stability**: Measure the fraction of time steps where $|\theta| < 0.1$ rad during evaluation roll-outs.

8. **Comparative Statistical Analysis**:
   - Compare convergence rates and final stabilization errors between Condition A and Condition B using confidence intervals from the 5 seeds.
   - Analyze if the structured critic with a misspecified $\alpha$ performs worse than the baseline to determine if the prior acts as a harmful bias when incorrect.