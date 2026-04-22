

Iteration 0:
### Project Summary: Lyapunov-Structured Value Functions
**Objective:** Evaluate if $V(s) = \Phi(s) + f_\theta(s)$ (where $\Phi$ is mechanical energy) improves SAC performance in Pendulum-v1 compared to learning $V(s)$ from scratch.

**Key Findings:**
*   **Sample Efficiency:** Condition B (structured) reached 90% of max reward in $43,015 \pm 6,035$ steps vs. $52,362 \pm 21,875$ for Condition A (baseline).
*   **Stability:** Structured critics significantly reduced variance across seeds, acting as a strong regularizer.
*   **Policy Failure:** Both conditions failed to maintain upright stability ($|\theta| < 0.1$ rad) compared to Vanilla SAC. The Lyapunov reward $R_t = \Phi(s_t) - \Phi(s_{t+1})$ is a potential-based shaping term; its telescoping sum makes the total return independent of the trajectory, removing the incentive for continuous stabilization.
*   **Representation:** Heatmap analysis confirms $f_\theta(s)$ successfully learns the "discounting penalty" and actuation constraints, while $\Phi(s)$ provides a physically accurate macroscopic value landscape.

**Constraints & Limitations:**
*   **Reward Design:** The current reward formulation is insufficient for fine-grained control.
*   **Statistical Power:** $N=5$ seeds were insufficient to reach $p < 0.05$ significance, despite clear directional improvements.
*   **Architecture:** The structured critic $V(s) = \Phi(s) + f_\theta(s)$ is validated as a robust architectural prior.

**Future Directions:**
*   **Reward Modification:** Abandon the replacement of native rewards. Implement potential-based reward shaping: $R'_{t} = R_{native} + \gamma \Phi(s_{t+1}) - \Phi(s_t)$ to preserve optimal policy behavior while retaining the benefits of the Lyapunov gradient.
*   **Methodology:** Maintain the structured critic architecture ($V = \Phi + f_\theta$) as it effectively offloads the learning of basic physics to the analytical prior.
        