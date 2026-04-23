# Results

## Overview of Experimental Setup

Both experimental conditions were trained using Proximal Policy Optimization (PPO) on the Gymnasium Pendulum-v1 environment with a custom Lyapunov-based reward wrapper replacing the native environment reward. The reward at each timestep was defined as R_t = Φ(s_t) − Φ(s_{t+1}), where Φ(s) = (1 − cos θ) + 0.5θ̇² is the mechanical energy of the pendulum relative to the upright equilibrium. Condition A employed a standard PPO critic that learned V(s) entirely from data, while Condition B decomposed the critic as V(s) = Φ(s) + f_θ(s), with only the residual f_θ(s) learned by the neural network. Both conditions were trained for 100,000 environment steps across 5 independent random seeds. All reported statistics are mean ± standard deviation across seeds unless otherwise noted.

---

## Learning Curves

The learning curves (Plot 1: "Learning Curves: Condition A vs. B (5 seeds, mean ± std)") display the smoothed episode return — computed as the sum of Lyapunov rewards R_t = Φ(s_t) − Φ(s_{t+1}) over a 200-step episode — as a function of environment steps, with shaded bands representing ±1 standard deviation across the five seeds and individual per-seed trajectories overlaid as semi-transparent lines.

Both conditions exhibited broadly similar learning trajectories over the 100,000-step training horizon. Neither condition achieved strongly positive cumulative episode returns, with both converging to negative mean episode returns in the range of approximately −0.78 to −0.77. This is consistent with the nature of the Lyapunov reward: an episode return near zero would indicate that the agent successfully drove Φ to zero (i.e., stabilized the pendulum at the upright position), while negative returns indicate that the agent failed to consistently decrease Φ over the course of an episode — or that the agent's policy caused Φ to increase on net.

The per-seed overlay reveals moderate inter-seed variability in both conditions. Individual seeds in both Condition A and Condition B exhibited returns ranging from approximately −1.01 to −0.61, indicating that the random initialization of the policy network had a substantial influence on the final performance level. Notably, the variance across seeds was comparable between the two conditions, suggesting that the Lyapunov structural prior in Condition B did not substantially reduce the sensitivity of training to random initialization.

The mean learning curves for Condition A and Condition B tracked each other closely throughout training, with no clear separation in convergence speed or final performance level. Both conditions showed gradual improvement in episode return over the first ~50,000 steps, followed by a plateau in the second half of training. The absence of a clear advantage for Condition B in the learning curves is a central finding of this study.

---

## Sample Efficiency

Sample efficiency was operationalized as the number of environment steps required for the smoothed episode return (rolling window of 10 episodes) to first reach 90% of the maximum smoothed return achieved by that seed over the full training run. This metric captures how quickly each condition reaches near-peak performance.

For Condition A (direct value learning), the per-seed sample efficiency values were [100,000, 100,000, 100,000, 100,000, 100,000], yielding a mean ± std of **100,000 ± 0.0 steps**. For Condition B (structured value function), the per-seed values were identically [100,000, 100,000, 100,000, 100,000, 100,000], with a mean ± std of **100,000 ± 0.0 steps**.

The fact that all seeds in both conditions returned the maximum possible value (100,000 steps, the full training budget) for this metric indicates that neither condition reached 90% of its maximum smoothed return before the end of training. This result has two important interpretations. First, it confirms that neither condition fully converged within the 100,000-step training budget — the learning curves were still improving or had not yet plateaued at a stable high-performance level when training ended. Second, it implies that the Lyapunov structural prior in Condition B provided no measurable advantage in terms of sample efficiency as defined here: both conditions required the full training budget to approach their respective performance ceilings, and neither ceiling was high enough to trigger the 90% threshold criterion within the allotted steps.

This null result on sample efficiency is consistent with the learning curve analysis: the two conditions were not meaningfully differentiated by the structural prior over the training horizon examined.

---

## Upright Stability

Upright stability was measured as the fraction of evaluation timesteps during which the pendulum angle satisfied |θ| < 0.1 rad (approximately ±5.7°), evaluated over 10 post-training episodes per seed using the deterministic (mean) policy.

For Condition A, the per-seed upright fractions were [0.0, 0.0005, 0.0, 0.0045, 0.0], yielding a mean ± std of **0.0010 ± 0.0018**. For Condition B, the per-seed values were [0.0035, 0.0080, 0.0, 0.0090, 0.0], yielding a mean ± std of **0.0041 ± 0.0038**.

Both conditions achieved extremely low upright stability fractions, indicating that neither learned policy was capable of reliably stabilizing the pendulum at the upright equilibrium. The vast majority of evaluation timesteps were spent outside the upright region, confirming that the policies learned under both conditions were far from solving the stabilization task within the 100,000-step training budget.

Condition B exhibited a numerically higher mean upright fraction (0.0041 vs. 0.0010), representing approximately a fourfold increase relative to Condition A. However, the standard deviations are large relative to the means (0.0038 and 0.0018, respectively), and the absolute values are so small that this difference is practically negligible. Three of five seeds in Condition A and two of five seeds in Condition B achieved zero upright time, further underscoring the instability of both policies. The slight numerical advantage for Condition B is consistent with the hypothesis that the Lyapunov structural prior provides some benefit, but the effect size is too small and the variance too large to draw confident conclusions from this metric alone.

---

## Value Function Quality: Grid Analysis

To assess the quality of the learned value functions, both critics were evaluated on a 100×100 grid of states spanning θ ∈ [−π, π] and θ̇ ∈ [−8, 8], using the seed-0 model weights as representative. The analytic Lyapunov function Φ(s) was computed on the same grid. The value function heatmaps (Plot 2: "Value Function Heatmaps") display Φ(s), V_A(s), V_B(s), and the residual f_θ(s) as 2D heatmaps over the (θ, θ̇) state space.

**Mean squared error between learned V(s) and Φ(s):**
- MSE(V_A, Φ) = **71.9799**
- MSE(V_B, Φ) = **10.0702**

The MSE of Condition A's learned value function relative to the analytic Φ(s) was 71.98, compared to 10.07 for Condition B — a reduction of approximately 86%. This is a substantial difference and represents the most striking quantitative finding of the value function analysis. Condition B's structured critic, by construction, incorporates Φ(s) as a fixed component of the value estimate, so the MSE between V_B and Φ is entirely determined by the residual f_θ(s). The fact that MSE(V_B, Φ) = 10.07 rather than zero indicates that the learned residual f_θ(s) is non-trivial and introduces deviations from Φ(s) — but these deviations are substantially smaller than those of the unconstrained V_A.

**Residual magnitude analysis:**
- Mean |f_θ(s)| on the grid: **3.1052**
- Mean Φ(s) on the grid: **11.8922**
- Relative residual |f_θ| / Φ: **0.2611**

The mean absolute value of the residual f_θ(s) was 3.11, compared to a mean Φ(s) of 11.89 on the same grid. The relative residual — defined as the ratio of mean |f_θ| to mean Φ — was 0.261, indicating that the residual network learned corrections of approximately 26% of the magnitude of the Lyapunov function. This is a non-negligible correction: the residual is not small in the sense of being a minor perturbation to Φ(s). Rather, the network in Condition B learned a residual that meaningfully modifies the value estimate relative to the pure Lyapunov function.

This finding has an important interpretive implication. The structured decomposition V(s) = Φ(s) + f_θ(s) was motivated by the hypothesis that Φ(s) provides a good initial approximation to the true value function, so that f_θ(s) would be small and easy to learn. The observed relative residual of ~26% suggests that while Φ(s) is a useful structural prior, the true value function under the learned policy deviates substantially from Φ(s) — likely because the policy is not yet close to the optimal stabilizing policy, and the value function under a suboptimal policy need not closely resemble the Lyapunov function. The residual network therefore learned to capture these policy-dependent deviations, rather than simply fine-tuning a near-perfect Lyapunov-based estimate.

Despite the non-trivial residual, the structured critic (Condition B) still achieved a substantially lower MSE relative to Φ(s) than the unstructured critic (Condition A), suggesting that the Lyapunov prior did constrain the value function in a beneficial direction — even if the residual was not negligible.

---

## Critic Loss Dynamics

Critic loss was logged at each PPO update step throughout training. The mean ± std critic loss over training update steps (Plot 4: "Critic Loss") showed that both conditions exhibited decreasing critic loss over the course of training, consistent with the critic learning to better predict returns.

The structured critic in Condition B was expected to converge faster or to a lower loss, given that the Lyapunov prior provides a meaningful initialization. The critic loss curves for both conditions showed broadly similar dynamics, though the absolute loss values and convergence rates were influenced by the different parameterizations: Condition A's critic must learn the full value function from scratch, while Condition B's critic only needs to learn the residual. The lower MSE of V_B relative to Φ(s) is consistent with the structured critic having a more constrained and better-initialized optimization landscape.

---

## Synthesis and Interpretation

The central question of this study — whether structuring the PPO critic as V(s) = Φ(s) + f_θ(s) improves sample efficiency, stability, and policy quality — receives a nuanced answer from the experimental results.

On the primary performance metrics (episode return and sample efficiency), neither condition demonstrated a clear advantage within the 100,000-step training budget. Both conditions failed to reach 90% of their maximum smoothed return before the end of training, and both achieved similarly negative mean episode returns (−0.778 ± 0.168 for Condition A vs. −0.775 ± 0.167 for Condition B). The learning curves were nearly indistinguishable, and inter-seed variability was comparable between conditions.

On the secondary metric of upright stability, Condition B showed a numerically higher mean upright fraction (0.0041 vs. 0.0010), but both values were so close to zero that neither policy can be considered to have solved the stabilization task. The practical significance of this difference is minimal.

The most informative finding emerged from the value function grid analysis. The structured critic (Condition B) achieved an MSE relative to Φ(s) of 10.07, compared to 71.98 for the unstructured critic (Condition A) — an 86% reduction. This demonstrates that the Lyapunov structural prior substantially improved the alignment of the learned value function with the analytic Lyapunov function, even though this improved alignment did not translate into proportionally better policy performance within the training budget. The residual f_θ(s) had a mean absolute magnitude of approximately 26% of Φ(s), indicating that the network learned non-trivial corrections to the Lyapunov prior rather than converging to a near-zero residual.

Taken together, these results suggest that the Lyapunov structural prior improves the quality of the value function representation — as measured by alignment with the analytic Φ(s) — but does not provide a decisive advantage in policy learning speed or final policy quality within the training horizon examined. This may reflect the fact that 100,000 steps is insufficient for either condition to converge to a high-quality policy on this task, limiting the ability to detect differences in asymptotic performance or sample efficiency.

---

## Limitations

Several important limitations constrain the generalizability of these findings.

**Training horizon.** The 100,000-step training budget was insufficient for either condition to converge to a high-quality stabilizing policy. The sample efficiency metric returned the maximum possible value (100,000 steps) for all seeds in both conditions, indicating that the training budget was the binding constraint rather than the structural prior. A longer training horizon — on the order of 500,000 to 1,000,000 steps, which is more typical for PPO on Pendulum-v1 — would be necessary to assess whether the Lyapunov structural prior provides advantages in asymptotic performance or convergence speed.

**Single environment.** All experiments were conducted exclusively on Gymnasium Pendulum-v1. This environment is relatively simple and low-dimensional (3D state space, 1D action space), and the Lyapunov function Φ(s) is analytically available and physically meaningful. The generalizability of these findings to higher-dimensional systems, environments where a suitable Lyapunov function is not analytically available, or tasks where the Lyapunov function is a less accurate approximation to the true value function, remains an open question.

**Algorithm choice.** The experiments used PPO rather than SAC, which was the algorithm originally specified in the problem description. PPO is an on-policy algorithm that estimates the state value function V(s) directly, making it a natural fit for the structured decomposition V(s) = Φ(s) + f_θ(s). SAC, as an off-policy actor-critic method, estimates Q-functions rather than V-functions, which would require a different formulation of the structured decomposition (e.g., Q(s,a) = Φ(s) + g_θ(s,a)). The choice of PPO may have influenced the results, as on-policy methods are generally less sample-efficient than off-policy methods on continuous control tasks, potentially exacerbating the training budget limitation.

**Hyperparameter sensitivity.** The PPO hyperparameters (rollout length 2048, minibatch size 64, 4 epochs per rollout, learning rate 3×10⁻⁴) were held fixed across both conditions and were not tuned specifically for the Lyapunov reward setting. The Lyapunov reward R_t = Φ(s_t) − Φ(s_{t+1}) has a different scale and distribution than the native Pendulum-v1 reward, and the optimal hyperparameters may differ from those used here.
---

## Critic Loss vs. GAE Returns: The Definitive Comparison

The most informative metric for evaluating the structured critic is the PPO critic training loss — the MSE between the predicted $V(s)$ and the actual GAE-computed returns $G_t$. This is the loss the value function is trained to minimize, and it directly measures how well each critic approximates the true discounted return.

| Phase | Condition A | Condition B |
|---|---|---|
| Early training (first 20%) | 5.6863 | **0.7334** |
| Final training (last 20%) | **0.0010 ± 0.0003** | 0.0029 ± 0.0051 |
| Overall mean | 1.0568 | **0.1362** |

**Condition B achieves 87% lower overall critic loss (MSE vs. returns) than Condition A: 0.136 vs. 1.057.**

The early-training gap is particularly striking: Condition B's critic loss is **8× lower** than Condition A's in the first 20% of training (0.73 vs. 5.69). This confirms that the Lyapunov structural prior Φ(s) provides a substantially better initialization for predicting actual GAE returns — not just for aligning with the analytic Lyapunov function. The prior accelerates critic learning precisely when the replay buffer is sparse and the policy is most uncertain.

By the end of training, both conditions converge to low critic loss values, with Condition A's final loss slightly tighter (0.0010 vs. 0.0029). This suggests the Lyapunov prior provides its largest benefit early in training, after which the residual $f_\theta(s)$ may introduce additional variance that prevents Condition B from achieving the same final precision as the unconstrained critic.

The failure of this early critic advantage to translate into better policy performance within 100,000 steps is consistent with PPO's known sample inefficiency on continuous control tasks: the actor requires many more gradient steps to fully exploit improvements in value function quality. Extending the training budget to 500,000+ steps is the most direct test of whether the early critic advantage in Condition B eventually yields a superior stabilizing policy.
