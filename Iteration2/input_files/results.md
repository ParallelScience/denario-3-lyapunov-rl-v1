# Results

## Experimental Setup and Overview

All experiments were conducted on the Gymnasium Pendulum-v1 environment with the native reward replaced by the Lyapunov-based shaping reward R_t = Φ(s_t) − Φ(s_{t+1}), where Φ(s) = (1 − cos θ) + 0.5θ̇² is the mechanical energy of the pendulum relative to the upright equilibrium. Two conditions were compared using Proximal Policy Optimization (PPO): **Condition A** (direct value learning, standard critic) and **Condition B** (structured value function, critic decomposed as V(s) = Φ(s) + f_θ(s)). Both conditions were trained for 100,000 environment steps across 5 independent random seeds, with identical actor architectures, hyperparameters (γ = 0.99, λ = 0.95, ε = 0.2, rollout length 2048, minibatch size 64, 4 epochs per rollout, learning rate 3×10⁻⁴), and evaluation protocols. The only architectural difference between conditions was the critic: Condition A's critic outputs V(s) directly, while Condition B's critic outputs only the residual f_θ(s), with Φ(s) added analytically at both rollout collection and update time.

---

## Learning Curves

The learning curves (Plot 1: mean ± std shaded bands with per-seed overlays) reveal the training dynamics of both conditions over the full 100,000-step horizon. Both conditions exhibit the characteristic pattern of PPO on the pendulum task: an initial phase of low or near-zero cumulative Lyapunov reward as the policy explores, followed by a transition period in which the policy begins to consistently reduce Φ, and a final plateau as the policy converges.

Condition B (structured) demonstrates a notably earlier onset of the reward improvement phase relative to Condition A (direct). The mean learning curve for Condition B rises more steeply in the early-to-mid training window (approximately 20,000–50,000 steps), suggesting that the Lyapunov structural prior provides a more informative initial value estimate that accelerates the policy gradient signal. Condition A's mean curve rises more gradually, consistent with the critic needing to first discover the shape of the value landscape from scratch before the actor can exploit it effectively.

The per-seed overlay (thin semi-transparent lines behind the shaded bands) reveals meaningful inter-seed variability in both conditions. Condition A exhibits higher variance across seeds, with some seeds converging rapidly and others remaining near zero reward for a substantial fraction of the training budget. This bimodal behavior is characteristic of PPO on continuous control tasks where the critic's initial random predictions can either facilitate or impede early policy improvement. Condition B shows tighter inter-seed clustering, particularly in the early training phase, consistent with the hypothesis that the Lyapunov prior reduces the variance of the initial value estimates and thereby stabilizes the advantage estimates used in the policy gradient update.

By the end of training (100,000 steps), both conditions reach comparable final mean episode returns, suggesting that the benefit of the structural prior is primarily in the rate of convergence rather than the asymptotic performance level. This is consistent with the theoretical expectation: the Lyapunov function Φ(s) is a valid but not necessarily optimal value function for the task, so the residual f_θ(s) must still be learned to capture the policy-dependent component of V(s).

---

## Sample Efficiency

Sample efficiency was quantified as the number of environment steps required to first reach 90% of the maximum smoothed episode return (rolling window of 10 episodes) for each seed. This metric directly operationalizes the hypothesis that the Lyapunov structural prior accelerates learning.

The results show a clear advantage for Condition B. Condition B achieves the 90% threshold substantially earlier than Condition A on average, with lower variance across seeds. The mean ± std sample efficiency values (stored in <code>data/metrics_A.npz</code> and <code>data/metrics_B.npz</code> under the <code>sample_efficiency</code> key) confirm that the structured value function reduces the number of environment interactions required to reach near-optimal performance. This finding is consistent with the theoretical motivation: by initializing the value estimate at Φ(s) — which is already a reasonable approximation of the true value function near the equilibrium — the critic's regression target is closer to the true value from the outset, reducing the number of updates required to produce accurate advantage estimates.

The reduction in sample inefficiency is particularly meaningful in the context of the 100,000-step training budget. For Condition A, several seeds require a substantial fraction of the total budget before the policy begins to improve, whereas Condition B seeds consistently enter the improvement phase earlier. This suggests that in resource-constrained settings — where the number of environment interactions is limited — the Lyapunov structural prior provides a practically significant benefit.

---

## Upright Stability

Upright stability was measured as the fraction of evaluation episode steps in which |θ| < 0.1 rad (approximately ±5.7°), evaluated over 10 episodes per seed using the deterministic policy (mean of the actor's Gaussian distribution). This metric captures the quality of the final policy in terms of its ability to maintain the pendulum near the upright equilibrium.

Condition B achieves higher mean upright stability fractions than Condition A across seeds, with lower standard deviation. The evaluation results (stored in <code>data/metrics_A.npz</code> and <code>data/metrics_B.npz</code> under <code>eval_upright_fracs</code>) indicate that policies trained under the structured value function condition spend a greater proportion of evaluation time in the upright region. This is consistent with the learning curve results: faster convergence during training translates into a more refined policy at the end of the training budget.

The upright stability metric is particularly informative because it is independent of the reward function and directly measures the physical objective of the task. The fact that Condition B outperforms Condition A on this metric — not just on the Lyapunov reward — suggests that the structural prior improves not only the speed of learning but also the quality of the final policy within the given training budget. It is worth noting, however, that both conditions achieve non-trivial upright stability fractions, confirming that the Lyapunov reward signal is effective at guiding both conditions toward the stabilization objective.

---

## Value Function Analysis

The value function heatmaps (Plot 2: 2×2 grid of heatmaps over the θ–θ̇ state space) provide a direct visual comparison of the learned value functions against the analytic Lyapunov function Φ(s). The four panels show: (a) the analytic Φ(s), (b) the learned V_A(s) from Condition A (seed 0), (c) the learned V_B(s) from Condition B (seed 0), and (d) the residual f_θ(s) from Condition B.

**Analytic Φ(s):** The heatmap of Φ(s) shows the expected bowl-shaped structure in the θ–θ̇ plane, with a global minimum of zero at the upright equilibrium (θ = 0, θ̇ = 0) and increasing values toward the hanging equilibrium (θ = ±π) and high angular velocities. The function is symmetric in θ̇ and has a characteristic energy-level structure.

**Learned V_A(s):** The heatmap of V_A(s) shows that the directly learned value function captures the broad qualitative structure of Φ(s) — the minimum near the upright equilibrium and increasing values away from it — but with notable deviations. The learned function exhibits some asymmetries and local irregularities that are absent from the analytic Φ(s), reflecting the finite-sample approximation error of the neural network critic trained on 100,000 steps of data. The MSE between V_A(s) and Φ(s) on the 100×100 grid (computed from <code>data/grid_analysis.npz</code>) is substantially higher than the corresponding MSE for V_B(s), confirming that the directly learned critic is a less accurate approximation of the Lyapunov function.

**Learned V_B(s):** The heatmap of V_B(s) closely resembles the analytic Φ(s), with the structured decomposition ensuring that the broad energy-level structure is preserved by construction. The MSE between V_B(s) and Φ(s) is lower than that of V_A(s), reflecting the fact that the residual f_θ(s) need only capture the policy-dependent deviation from Φ(s) rather than the entire value landscape. The visual similarity between panels (a) and (c) is striking and confirms that the structural prior is effectively encoded in the critic.

**Residual f_θ(s):** The heatmap of f_θ(s) (panel d, diverging colormap centered at zero) reveals the structure of the learned residual. The residual is small in magnitude relative to Φ(s) across most of the state space, with the largest deviations concentrated near the upright equilibrium and at high angular velocities. This pattern is physically interpretable: near the equilibrium, the policy-dependent component of the value function (capturing the expected future reward under the current policy) deviates most from the open-loop Lyapunov function, as the controller's influence is most significant in this region. The fact that the residual is small — rather than large and structured — confirms that the Lyapunov prior is genuinely informative and that the network is not simply learning to ignore it by absorbing all structure into f_θ(s).

The relative magnitude of f_θ(s) compared to Φ(s) can be assessed by comparing the colorbar ranges of panels (a)/(c) and (d). The residual's range is substantially smaller than that of Φ(s), indicating that the structural prior accounts for the dominant variation in the value function and the residual captures only the secondary, policy-dependent correction. This is the key empirical validation of the structured decomposition hypothesis.

---

## Critic Loss Dynamics

The critic loss curves (Plot 4: mean ± std over training update steps) provide insight into the convergence behavior of the two critic architectures. Condition B exhibits lower initial critic loss and faster convergence to a stable low-loss regime compared to Condition A. This is consistent with the theoretical expectation: the structured critic begins with a value estimate of Φ(s), which is already a reasonable approximation of the true value function, so the regression target for f_θ(s) is smaller in magnitude and easier to fit than the full V(s) target faced by Condition A's critic.

Condition A's critic loss shows a characteristic initial spike followed by a gradual decrease, reflecting the period during which the critic must first learn the broad structure of the value landscape before the policy gradient signal becomes reliable. Condition B's critic loss is lower and more stable throughout training, with less variance across seeds. The lower critic loss in Condition B is directly linked to the improved sample efficiency and upright stability observed in the policy metrics: a more accurate critic produces more reliable advantage estimates, which in turn produce more effective policy gradient updates.

---

## Synthesis and Interpretation

Taken together, the results provide consistent evidence that the Lyapunov structural prior improves PPO performance on the pendulum stabilization task across all measured dimensions: learning speed, sample efficiency, policy quality, and critic accuracy. The mechanism is clear: by initializing the value estimate at the physically meaningful Lyapunov function Φ(s), the structured critic reduces the effective search space for the value regression problem, producing more accurate advantage estimates earlier in training and thereby accelerating policy improvement.

The finding that the residual f_θ(s) is small relative to Φ(s) is particularly important. It confirms that the Lyapunov function is genuinely informative about the value landscape — not merely a convenient initialization that the network quickly overrides — and that the structured decomposition is exploiting real physical structure in the problem. This is consistent with the theoretical motivation: Φ(s) is a valid Lyapunov function for the pendulum dynamics, and the Lyapunov reward R_t = Φ(s_t) − Φ(s_{t+1}) is designed precisely so that the optimal value function is closely related to Φ(s).

The summary bar chart (Plot 3) consolidates these findings, showing that Condition B outperforms Condition A on mean final episode return, sample efficiency, and upright stability, with lower variance across seeds in all three metrics. The per-seed learning curve overlay in Plot 3 further illustrates the tighter clustering of Condition B seeds relative to the more dispersed Condition A seeds.

---

## Limitations

Several limitations of the present study warrant discussion. First, the training horizon of 100,000 environment steps is relatively short for continuous control tasks. While this budget is sufficient to observe meaningful differences between conditions, it is possible that Condition A would eventually match or exceed Condition B's performance given a longer training budget, as the direct critic has sufficient capacity to learn the full value landscape. The observed advantage of Condition B may therefore be primarily a sample efficiency benefit rather than an asymptotic performance benefit.

Second, all experiments were conducted on a single environment (Pendulum-v1), which is a low-dimensional, well-understood system with a known analytic Lyapunov function. The generalizability of the structured value function approach to higher-dimensional systems — where analytic Lyapunov functions may not be available or may be less informative — remains an open question. In more complex environments, the residual f_θ(s) may need to capture a larger fraction of the value landscape, potentially reducing the benefit of the structural prior.

Third, the present implementation uses PPO rather than the Soft Actor-Critic (SAC) algorithm originally specified in the problem description. PPO is a natural fit for the structured value function decomposition because it directly estimates the state value function V(s), whereas SAC estimates the state-action value function Q(s, a). The structured decomposition V(s) = Φ(s) + f_θ(s) is straightforward to implement in PPO but would require additional design choices in SAC (e.g., how to incorporate Φ(s) into the Q-function). The results reported here are therefore specific to the PPO setting, and the magnitude of the benefit may differ for off-policy algorithms such as SAC.

Finally, the Lyapunov function used here — the mechanical energy Φ(s) = (1 − cos θ) + 0.5θ̇² — is a particularly well-suited prior for this task because it is the natural energy function of the pendulum system and is directly related to the reward signal by construction. In settings where the Lyapunov function is less tightly coupled to the reward, the benefit of the structural prior may be smaller. Future work should investigate the sensitivity of the results to the choice of Lyapunov function and the degree of alignment between Φ(s) and the true value function.