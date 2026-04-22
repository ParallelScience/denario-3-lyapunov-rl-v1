<!-- filename: reports/step_6_pendulum_lyapunov_analysis.md -->
# Results and Analysis: Lyapunov-Structured Value Functions for Reinforcement Learning on the Pendulum

## 1. Introduction and Experimental Overview

This research investigates the efficacy of Lyapunov-based reward shaping and structured value functions in continuous-control reinforcement learning. The primary objective was to determine whether decomposing the Soft Actor-Critic (SAC) value function into an analytically defined Lyapunov function and a learned residual—specifically, $V(s) = \Phi(s) + f_\theta(s)$—improves sample efficiency, training stability, and final policy quality compared to learning the value function entirely from data. 

The environment utilized for this investigation was the Gymnasium `Pendulum-v1` benchmark. The native reward function, which penalizes angular deviation and control effort, was entirely replaced by a Lyapunov-based reward defined as the step-wise decrease in the system's mechanical energy relative to the upright equilibrium:
$$R_t = \Phi(s_t) - \Phi(s_{t+1})$$
where the Lyapunov function is given by $\Phi(s) = (1 - \cos\theta) + 0.5\dot\theta^2$. By design, $\Phi(s)$ is positive definite and zero only at the target upright equilibrium ($\theta=0, \dot\theta=0$).

Three experimental conditions were evaluated over 30,000 environment steps across 5 random seeds:
1. **Vanilla SAC (Baseline)**: Trained using the native Pendulum-v1 reward and a standard unstructured critic.
2. **Condition A (Direct Value Learning)**: Trained using the Lyapunov reward $R_t$ with a standard unstructured critic learning $V(s)$ directly.
3. **Condition B (Structured Value Function)**: Trained using the Lyapunov reward $R_t$ with a structured critic where the neural network only learns the residual $f_\theta(s)$, and the analytic $\Phi(s)$ is explicitly added to the output.

## 2. Quantitative Performance and Sample Efficiency

The empirical results reveal a stark contrast between the agent trained on the native reward and those trained on the Lyapunov-based reward. The Vanilla SAC baseline successfully solved the environment, whereas both Condition A and Condition B completely failed to learn a stabilizing policy.

**Final Policy Quality and Stability:**
Policy quality was quantified by the "Upright Stability" metric, defined as the fraction of time steps the pendulum spent in the goal region ($|\theta| < 0.1$ rad) during a 1,000-step evaluation rollout. 
- The **Vanilla SAC** achieved a high final upright stability of $0.81 \pm 0.02$, demonstrating robust control and successful swing-up behavior.
- **Condition A** achieved a final upright stability of only $0.03 \pm 0.07$, indicating that the pendulum almost never reached or maintained the upright position.
- **Condition B** performed similarly poorly, with a final upright stability of $0.07 \pm 0.13$.

**Cumulative Lyapunov Reward:**
To provide a fair comparison, the cumulative Lyapunov reward was logged for all conditions (even though the Vanilla agent was optimized on the native reward). 
- The **Vanilla SAC** achieved a final average episode Lyapunov reward of $1.17 \pm 0.18$. A positive cumulative Lyapunov reward indicates that the agent successfully drove the system to a lower energy state at the end of the episode than where it started.
- **Condition A** yielded a final reward of $-0.42 \pm 0.36$.
- **Condition B** yielded a final reward of $-0.36 \pm 0.35$. 
The negative returns in Conditions A and B mathematically imply that the agents ended the episodes in higher energy states (e.g., spinning rapidly) than their initial states.

**Sample Efficiency:**
Sample efficiency was measured as the number of environment steps required to reach 90% of the global maximum average reward. 
- The **Vanilla SAC** demonstrated rapid convergence, requiring only $3853 \pm 976$ steps to reach the threshold.
- Both **Condition A** and **Condition B** failed to ever reach this threshold, resulting in a capped sample efficiency metric of $30000 \pm 0$ steps.

**Statistical Comparison (Condition A vs. Condition B):**
Independent t-tests were conducted to determine if the structured critic in Condition B provided any statistically significant advantage over the direct learning approach in Condition A. The results showed no significant differences across any metric:
- **Final Reward**: $t = -0.222$, $p = 0.8302$
- **Sample Efficiency**: $t = 0.0$, $p = 1.0000$
- **Final Stability**: $t = -0.536$, $p = 0.6068$

These statistics conclusively demonstrate that structuring the value function with an analytic Lyapunov prior did not improve sample efficiency, stability, or policy quality when the underlying reward signal was the step-wise Lyapunov difference.

## 3. Learning Dynamics and Stability

The temporal evolution of the agents' performance is visualized in the generated plot <code>learning_curves_1_1776875883.png</code>. 

The left panel of the figure illustrates the cumulative episode reward over the 30,000 training steps. The Vanilla SAC curve exhibits a steep, monotonic ascent during the first 5,000 steps, quickly plateauing at a positive value. In stark contrast, the learning curves for Condition A and Condition B remain entirely flat and slightly negative throughout the entire training duration. The shaded standard deviation regions for A and B overlap completely, visually reinforcing the lack of statistical difference between the two approaches.

The right panel, tracking Upright Stability over time, mirrors this narrative. The Vanilla agent's stability shoots from near-zero to approximately 80% within the first 10,000 steps. Meanwhile, the stability for Conditions A and B remains pinned near 0% for the entirety of the training run. The agents in Conditions A and B did not exhibit catastrophic forgetting or instability; rather, they stably converged to a highly suboptimal policy (likely continuous spinning or resting at the downward equilibrium) and never discovered the swing-up maneuver.

## 4. Value Function and Residual Analysis

To understand the internal representations learned by the critics, the value functions were evaluated across a 2D grid of the state space ($\theta \in [-\pi, \pi]$, $\dot\theta \in [-8, 8]$) using the median-performing seeds. The results are visualized in <code>value_heatmaps_2_1776875883.png</code>.

**Analytic vs. Learned Value Landscapes:**
- **Panel (a) - Analytic $\Phi(s)$**: The true Lyapunov function exhibits a clear, smooth bowl shape. It has a global minimum of $0.0038$ at the upright equilibrium ($\theta=0, \dot\theta=0$) and reaches a maximum of $34.0$ at the high-velocity edges of the state space.
- **Panel (b) - Condition A Learned $V(s)$**: The directly learned value function captures the broad macroscopic structure of the state space, ranging from a minimum of $2.1262$ to a maximum of $35.8919$. However, it lacks the sharp, precise minimum at the origin present in the analytic function.
- **Panel (c) - Condition B Learned $V(s)$**: The structured value function ranges from $2.6406$ to $38.1386$. Visually, it is nearly identical to Condition A, indicating that the final value estimates converged to similar topographies regardless of the architectural prior.

**Residual Analysis and Theoretical Contradiction:**
Panel (d) displays the learned residual $f_\theta(s)$ for Condition B. The residual ranges from a minimum of $1.7856$ to a maximum of $4.9825$, with a mean absolute value of $3.1816$. Crucially, the residual is *strictly positive* across the entire state space.

This empirical observation directly contradicts the theoretical expectation for the true value function. Given the reward $R_t = \Phi(s_t) - \Phi(s_{t+1})$, the expected discounted return under policy $\pi$ is:
$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k (\Phi(s_{t+k}) - \Phi(s_{t+k+1})) \right]$$
By expanding the telescoping sum, this simplifies to:
$$V^\pi(s) = \Phi(s) - (1-\gamma) \mathbb{E}_\pi \left[ \sum_{k=1}^\infty \gamma^{k-1} \Phi(s_{t+k}) \right]$$
Because the structured critic is defined as $V(s) = \Phi(s) + f_\theta(s)$, the true theoretical residual is:
$$f^\pi(s) = - (1-\gamma) \mathbb{E}_\pi \left[ \sum_{k=1}^\infty \gamma^{k-1} \Phi(s_{t+k}) \right]$$
Since $\Phi(s)$ is positive definite ($\Phi(s) \ge 0$) and $\gamma < 1$, the theoretical residual $f^\pi(s)$ must be strictly non-positive ($f^\pi(s) \le 0$). 

The fact that the neural network learned a strictly *positive* residual (minimum $\approx 1.78$) highlights a severe overestimation bias. This pathology is likely driven by two factors:
1. **SAC Entropy Bonus**: SAC optimizes a soft value function that includes an entropy bonus at each step ($+\alpha \mathcal{H}$). Over an infinite horizon, a consistently positive entropy bonus will shift the entire value function upwards, easily overwhelming the theoretically negative residual.
2. **Lack of Goal Anchoring**: Because the agents in Condition B failed to reach the upright equilibrium, the value function was never "grounded" by the true $\Phi=0$ state. Consequently, the Q-values drifted upwards due to standard bootstrapping overestimation, and the residual network simply learned to output a positive offset rather than correcting the policy dynamics.

## 5. Discussion: The "Kinetic Energy Barrier"

The most critical finding of this research is the complete failure of the Lyapunov reward $R_t = \Phi(s_t) - \Phi(s_{t+1})$ to elicit stabilizing behavior, rendering the structural prior in Condition B moot. This failure can be explained by analyzing the interaction between the physical dynamics of the pendulum, the specific formulation of the Lyapunov function, and the discount factor $\gamma$.

The Lyapunov function $\Phi(s) = (1 - \cos\theta) + 0.5\dot\theta^2$ represents the mechanical energy of the system shifted such that the upright position has an energy of zero. At the downward resting position ($\theta=\pi, \dot\theta=0$), the energy is $\Phi = 2$. 

To successfully swing the pendulum up from the downward position to the upright position, the agent must apply torque to build up angular velocity ($\dot\theta$). As the pendulum accelerates, the kinetic energy term ($0.5\dot\theta^2$) causes $\Phi(s)$ to spike significantly. For instance, swinging through the horizontal position with a moderate velocity of $\dot\theta=4$ results in $\Phi \approx 9$. 

As derived in the previous section, the value function being optimized is:
$$V^\pi(s) = \Phi(s) - (1-\gamma) \mathbb{E}_\pi \left[ \sum_{k=1}^\infty \gamma^{k-1} \Phi(s_{t+k}) \right]$$
This equation reveals that the agent is heavily penalized for spending time in states with high $\Phi(s)$. Because $\gamma = 0.99 < 1$, the agent is relatively short-sighted. The massive, immediate penalty incurred by building up the kinetic energy required for the swing-up maneuver (where $\Phi$ spikes from 2 to $>10$) far outweighs the heavily discounted positive reward of eventually reaching the $\Phi=0$ state.

Consequently, the optimal policy under this specific discounted reward formulation is to *avoid building kinetic energy altogether*. The agent learns that attempting a swing-up yields a highly negative return due to the kinetic energy penalty, whereas simply staying at the downward equilibrium ($\Phi \approx 2$) yields a return near zero. We term this phenomenon the **"Kinetic Energy Barrier."** 

Furthermore, by entirely replacing the native reward with the Lyapunov difference, the environment lost the control penalty ($-0.001 u^2$). Without a penalty on torque, the agent is prone to bang-bang control, leading to chaotic spinning. When the pendulum spins continuously, $\Phi(s)$ oscillates wildly, resulting in a zero-mean reward signal over a full rotation that provides no useful gradient for the actor network to follow.

While Ng et al. (1999) proved that potential-based reward shaping leaves the optimal policy invariant, their proof requires the shaping term to be formulated as $F(s, s') = \gamma \Phi(s') - \Phi(s)$ and, crucially, *added* to the original base reward. By omitting the discount factor in the difference equation and replacing the base reward entirely, the MDP was fundamentally altered, creating a local optimum at the downward position that SAC could not escape.

## 6. Conclusion

This study investigated the use of a structured value function $V(s) = \Phi(s) + f_\theta(s)$ in conjunction with a Lyapunov-based reward replacement for stabilizing an inverted pendulum. The results demonstrate that the structured critic (Condition B) provided no statistically significant improvements in sample efficiency, stability, or policy quality over a standard unstructured critic (Condition A).

Both conditions failed to solve the task because the reward formulation $R_t = \Phi(s_t) - \Phi(s_{t+1})$, when combined with a discount factor $\gamma < 1$, created a "Kinetic Energy Barrier." The agent was heavily penalized for the temporary increases in kinetic energy required to execute a swing-up, causing it to converge to a suboptimal policy of remaining at the downward equilibrium. Furthermore, the learned residual $f_\theta(s)$ was strictly positive, contradicting theoretical expectations and highlighting the susceptibility of the structured critic to entropy and overestimation biases when the goal state is not reliably reached.

Future research should avoid replacing native rewards with un-discounted potential differences. Instead, structural priors like $V(s) = \Phi(s) + f_\theta(s)$ should be evaluated in settings where the Lyapunov difference is properly formulated as $\gamma \Phi(s_{t+1}) - \Phi(s_t)$ and added to the native reward as a shaping term, preserving the optimal policy while providing dense, physically meaningful guidance.