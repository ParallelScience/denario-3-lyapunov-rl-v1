# Results and Analysis: Lyapunov-Structured Value Functions for Reinforcement Learning

## 1. Executive Summary
This study investigates the efficacy of Lyapunov-based reward shaping and structured value functions in continuous-control reinforcement learning. Using the Gymnasium `Pendulum-v1` environment, we compared three experimental conditions: a Vanilla Soft Actor-Critic (SAC) baseline trained on the native environment reward, Condition A (Direct Value Learning) trained on a Lyapunov difference reward $R_t = \Phi(s_t) - \Phi(s_{t+1})$, and Condition B (Structured Value Function) trained on the same Lyapunov reward but utilizing a decomposed critic architecture $Q(s,a) = \Phi(s) + f_\theta(s,a)$. 

The empirical results reveal a striking failure of the Lyapunov difference reward to induce stabilizing behaviors in both Condition A and Condition B, in stark contrast to the successful Vanilla baseline. Through rigorous mathematical analysis of the Bellman equations and the discounted return, we demonstrate that replacing the native reward entirely with a potential difference inadvertently scales down the action-dependent learning signal by a factor of $(1-\gamma)$. While the structured critic in Condition B successfully isolates this residual signal, the fundamentally diminished magnitude of the effective reward prevents the policy from overcoming the entropy regularization inherent to SAC. This report details the quantitative findings, provides a comprehensive theoretical framework explaining the failure modes, and analyzes the learned value function heatmaps to corroborate the mathematical derivations.

## 2. Quantitative Performance and Sample Efficiency
Training was conducted for 30,000 environment steps across 5 random seeds for each condition. To ensure a fair comparison, the Vanilla SAC baseline was trained using the native Pendulum reward but evaluated using the Lyapunov reward metric. 

The performance metrics clearly delineate the success of the baseline and the failure of the experimental conditions:
*   **Vanilla SAC (Baseline):** The agent successfully learned to stabilize the pendulum, achieving a sample efficiency (steps to reach 90% of the maximum global reward) of $7625.42 \pm 1371.32$ steps. The final upright stability—defined as the fraction of time steps where $|\theta| < 0.1$ rad—reached $0.6379 \pm 0.2053$. The final average Lyapunov reward per episode was $1.17 \pm 0.18$.
*   **Condition A (Direct Value Learning):** The agent failed to learn a stabilizing policy. The sample efficiency metric defaulted to the maximum $30000.0 \pm 0.0$ steps, as it never reached the 90% threshold. Final upright stability was negligible at $0.0335 \pm 0.0671$, and the final average reward was negative ($-0.42 \pm 0.36$), indicating that the pendulum ended episodes with more mechanical energy than it started with.
*   **Condition B (Structured Value Function):** Despite the architectural inductive bias, Condition B performed identically to Condition A. Sample efficiency was $30000.0 \pm 0.0$ steps, final upright stability was $0.0438 \pm 0.0536$, and the final average reward was $-0.36 \pm 0.35$.

Statistical comparisons between Condition A and Condition B confirm the absence of any significant performance difference. An independent t-test on final upright stability yielded $t = -0.2391$ ($p = 0.8170$), and a t-test on final average reward yielded $t = -0.2215$ ($p = 0.8302$). 

These dynamics are visually corroborated in the generated plot `step_5_learning_curves_stability_1_1776874625.png`. The learning curves demonstrate that while the Vanilla baseline rapidly ascends to a positive episodic return, both Condition A and Condition B remain entirely flat and negative throughout the 30,000 training steps. The upright stability sub-plot mirrors this trajectory, with the baseline climbing to over 80% stability during periodic evaluations, while the experimental conditions remain pinned near zero.

## 3. Theoretical Analysis of the Reward Formulation
The catastrophic failure of both Condition A and Condition B necessitates a deeper theoretical examination of the reward formulation. The core issue lies in the fundamental mechanics of potential-based reward shaping in discounted Markov Decision Processes (MDPs).

The experimental reward was defined as the undiscounted difference in the Lyapunov function:
$$R_t = \Phi(s_t) - \Phi(s_{t+1})$$

In a reinforcement learning setting with a discount factor $\gamma \in (0, 1)$, the agent seeks to maximize the expected discounted return $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k}$. Substituting our reward definition into the return yields a telescoping series that does not perfectly cancel:
$$G_t = (\Phi(s_t) - \Phi(s_{t+1})) + \gamma (\Phi(s_{t+1}) - \Phi(s_{t+2})) + \gamma^2 (\Phi(s_{t+2}) - \Phi(s_{t+3})) + \dots$$
$$G_t = \Phi(s_t) - (1-\gamma)\Phi(s_{t+1}) - \gamma(1-\gamma)\Phi(s_{t+2}) - \dots$$
$$G_t = \Phi(s_t) - (1-\gamma) \sum_{k=1}^\infty \gamma^{k-1} \Phi(s_{t+k})$$

Taking the expectation over the policy $\pi$, the true action-value function $Q^\pi(s_t, a_t)$ becomes:
$$Q^\pi(s_t, a_t) = \Phi(s_t) - (1-\gamma) \mathbb{E}^\pi \left[ \sum_{k=1}^\infty \gamma^{k-1} \Phi(s_{t+k}) \Big| s_t, a_t \right]$$

This derivation reveals a profound structural flaw in the experimental design. According to the theory of potential-based reward shaping, a shaping reward is designed to be *added* to a native task reward. By replacing the native reward entirely with the potential difference, we effectively set the underlying task reward to zero. Consequently, the advantage function—the signal that drives policy improvement—is artificially scaled down by a factor of $(1-\gamma)$. 

In our experiments, $\gamma = 0.99$, meaning $(1-\gamma) = 0.01$. The state-dependent baseline $\Phi(s_t)$ dominates the Q-value, ranging from $0$ at the equilibrium to approximately $34$ at the maximum energy state. In contrast, the action-dependent variation (the second term in the equation) is scaled down by a factor of $100$. 

In Condition A, the standard neural network critic $Q_\theta(s,a)$ must approximate this function from scratch. The mean squared error loss is overwhelmingly dominated by the need to fit the large, state-dependent $\Phi(s_t)$ baseline. The tiny action-dependent variations are entirely lost in the neural network's approximation noise. As a result, the policy gradient $\nabla_a Q_\theta(s,a)$ consists mostly of noise, preventing the actor from learning a stabilizing policy.

## 4. Analysis of the Structured Critic (Condition B)
Condition B was designed to mitigate the approximation issues of Condition A by explicitly structuring the critic as $Q(s,a) = \Phi(s) + f_\theta(s,a)$. By injecting the analytic Lyapunov function, the network $f_\theta$ is theoretically freed from learning the large state-dependent baseline. 

We can derive the exact target that the residual network $f_\theta$ is forced to learn by substituting the structured Q-function into the standard Bellman residual $\delta_t$:
$$\delta_t = R_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$$
$$\delta_t = (\Phi(s_t) - \Phi(s_{t+1})) + \gamma (\Phi(s_{t+1}) + f_\theta(s_{t+1}, a_{t+1})) - (\Phi(s_t) + f_\theta(s_t, a_t))$$
$$\delta_t = -(1-\gamma)\Phi(s_{t+1}) + \gamma f_\theta(s_{t+1}, a_{t+1}) - f_\theta(s_t, a_t)$$

Setting the expected Bellman residual to zero shows that $f_\theta(s,a)$ is exactly the action-value function for a surrogate MDP where the reward is $\tilde{R}_t = -(1-\gamma)\Phi(s_{t+1})$. 

While the structural decomposition elegantly isolates the action-dependent component, it does not solve the problem of the diminished reward scale. The surrogate reward $\tilde{R}_t$ is bounded between approximately $-0.34$ and $0$. Consequently, the gradients of the residual $\nabla_a f_\theta(s,a)$ are exceptionally small. 

In the Soft Actor-Critic algorithm, the policy is updated to maximize $Q(s,a) - \alpha \log \pi(a|s)$. Because $\nabla_a Q(s,a) = \nabla_a f_\theta(s,a)$, the policy gradient is driven entirely by the residual. Given the microscopic scale of the residual gradients, the entropy maximization term $\alpha \log \pi(a|s)$ overwhelmingly dominates the policy loss. Even with automatic entropy tuning enabled, the adaptation of the temperature parameter $\alpha$ is likely too slow, or the signal-to-noise ratio of the gradients is too poor, to prevent the agent from collapsing into a purely random, high-entropy policy. This explains why Condition B failed just as completely as Condition A.

## 5. Visual Analysis of the Value Functions and Residuals
To empirically validate our theoretical derivations, we analyzed the learned value functions across a 2D grid of the state space ($\theta \in [-\pi, \pi]$, $\dot{\theta} \in [-8, 8]$). The resulting visualizations are documented in `step_5_value_function_heatmaps_2_1776874625.png`.

*   **Panel (a) - Analytic Lyapunov Function $\Phi(s)$:** This panel displays the ground-truth mechanical energy of the pendulum. It exhibits the expected positive-definite bowl shape, with the global minimum of $0$ located precisely at the upright equilibrium ($\theta=0, \dot{\theta}=0$) and maximum values at the boundaries of the velocity space.
*   **Panel (b) - Condition A Learned Value Function $V(s)$:** The heatmap for Condition A shows a distorted, noisy approximation of the state space. Because the policy failed to stabilize and acts randomly, the learned Q-values reflect the high expected energy of a chaotic trajectory rather than the sharp, well-defined structure of an optimal value function. The lack of precise gradients in this landscape visually confirms why the actor could not extract a stabilizing policy.
*   **Panel (c) - Condition B Learned Value Function $V(s)$:** In contrast to Condition A, the total learned value function for Condition B visually resembles the analytic $\Phi(s)$ from Panel (a). This is because the total value is defined as $\Phi(s) + f_\theta(s,a)$, and as proven in Section 4, the magnitude of $f_\theta$ is constrained to be roughly 100 times smaller than $\Phi(s)$. Thus, the analytic prior dominates the visual representation.
*   **Panel (d) - Condition B Learned Residual $f_\theta(s)$:** This panel is the most revealing. Plotted using a diverging colormap, the residual $f_\theta(s)$ is shown to be strictly negative across the entire state space. This perfectly corroborates our mathematical proof that $f_\theta$ learns the discounted future energy penalty $-(1-\gamma) \mathbb{E}[\sum \gamma^{k-1} \Phi(s_{t+k})]$. The network successfully captured the structure of the surrogate reward, assigning the largest negative penalties to regions of the state space where the random policy accumulates the most future energy (e.g., high velocities). The fact that the residual network successfully learned this structure—yet the policy still failed—definitively proves that the bottleneck was the *scale* of the gradients relative to the entropy penalty, rather than the *representational capacity* of the critic.

## 6. Implications and Recommendations for Future Work
The central hypothesis of this research was that initializing a value estimate with a physically meaningful Lyapunov function would reduce the residual search space, thereby improving sample efficiency and stability. The empirical results reject this hypothesis under the current experimental design, but the theoretical analysis provides a clear explanation for the failure and offers actionable pathways for future research.

The fundamental error was replacing the native task reward entirely with the undiscounted Lyapunov difference $R_t = \Phi(s_t) - \Phi(s_{t+1})$. This formulation inadvertently strips the MDP of its primary learning signal, leaving only a tiny artifact scaled by $(1-\gamma)$ to guide the policy. While the structural decomposition $V(s) = \Phi(s) + f_\theta(s)$ is mathematically sound and successfully isolates this residual, it cannot magically restore the magnitude of a fundamentally diminished reward signal in the presence of entropy regularization.

To successfully leverage Lyapunov-based reward shaping and structured critics in future applications, we recommend the following modifications:

1.  **Reward Scaling:** If the Lyapunov difference is to be used as the sole reward, it must be scaled by $\frac{1}{1-\gamma}$. Defining the reward as $R_t = \frac{1}{1-\gamma} (\Phi(s_t) - \Phi(s_{t+1}))$ would restore the surrogate reward to $-\Phi(s_{t+1})$, ensuring that the action-dependent gradients have a magnitude comparable to standard RL tasks, preventing entropy domination.
2.  **Additive Potential-Based Shaping:** Rather than replacing the native reward, the Lyapunov function should be used as an additive shaping term in accordance with standard potential-based shaping theory: $R'_t = R_{native} + \gamma \Phi(s_{t+1}) - \Phi(s_t)$. This preserves the original task specification (including vital control effort penalties) while providing dense, theoretically grounded guidance toward the equilibrium.
3.  **Inclusion of Control Penalties:** The Lyapunov difference reward solely penalizes state deviations. Without a penalty for control effort (such as the $-0.001 u^2$ term in the native Pendulum reward), the optimal policy is encouraged to utilize aggressive, bang-bang control. Such policies are notoriously difficult for neural networks to learn and execute stably in discrete-time environments, further contributing to the observed instability.

## 7. Conclusion
This study highlights a critical intersection between reward design, algorithmic regularization, and network architecture. While embedding analytic priors into neural network critics via $V(s) = \Phi(s) + f_\theta(s)$ is a powerful technique for isolating action-dependent advantages, it is highly sensitive to the mathematical formulation of the reward. The failure of the Lyapunov difference reward in this environment serves as a rigorous case study demonstrating that potential differences cannot be used as standalone rewards in discounted RL without careful attention to scaling factors and gradient magnitudes. Implementing the recommended scaling and additive shaping techniques will be essential for realizing the theoretical benefits of Lyapunov-structured value functions in future continuous-control research.