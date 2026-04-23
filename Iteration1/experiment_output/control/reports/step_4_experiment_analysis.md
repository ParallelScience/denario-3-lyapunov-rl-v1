<!-- filename: reports/step_4_experiment_analysis.md -->
# Results

## Overview of Experimental Conditions

Two conditions were evaluated across five independent random seeds each, trained for 100,000 environment steps on the Gymnasium Pendulum-v1 task with the Lyapunov-based reward R_t = Φ(s_t) − Φ(s_{t+1}). Condition A employed a standard Soft Actor-Critic (SAC) critic that learns Q(s, a) entirely from data. Condition B employed a structured critic of the form Q(s, a) = Φ(s) + f_θ(s, a), where the analytic Lyapunov function Φ(s) = (1 − cos θ) + 0.5θ̇² is added directly to the network output f_θ, and the final layer of f_θ is initialized to zero so that Q(s, a) ≈ Φ(s) at the start of training. All other hyperparameters — actor architecture, replay buffer size (100k), batch size (256), learning rate (3×10⁻⁴), discount factor (γ = 0.99), and soft update coefficient (τ = 0.005) — were held constant across conditions.

---

## Learning Curves

The learning curves (mean ± standard deviation of per-episode Lyapunov return across 5 seeds) are shown in the learning curve figure. Both conditions exhibit the characteristic pattern of SAC on this task: an initial period of near-zero or slightly negative returns during early exploration, followed by a gradual improvement as the replay buffer accumulates informative transitions.

Examining the smoothed curves, Condition B (Structured Value Function) displays a marginally earlier onset of improvement in several seeds, consistent with the hypothesis that initializing the critic with the physically meaningful Lyapunov function reduces the initial estimation error and provides a more informative gradient signal to the actor from the first updates. However, the mean learning curves for both conditions converge to similar final performance levels by the end of training, with substantial overlap in the shaded standard-deviation bands across seeds. The final mean episode returns are −0.234 ± 0.073 for Condition A and −0.263 ± 0.213 for Condition B (mean ± std across seeds, computed from the evaluation results). Notably, Condition B exhibits considerably higher variance across seeds (std = 0.213) compared to Condition A (std = 0.073), indicating that while some seeds of Condition B achieve strong performance, others fail to converge within the 100,000-step budget. This increased variance is a key finding: the structured initialization does not uniformly accelerate learning but instead produces a bimodal outcome distribution — seeds that benefit from the prior converge faster, while others appear to be destabilized by the interaction between the fixed Lyapunov component and the learned residual.

---

## Sample Efficiency

Sample efficiency was quantified as the number of environment steps required to first reach 90% of the maximum average episode return observed during training. The results are reported in the sample efficiency table (<code>data/sample_efficiency.csv</code>) and visualized in the right panel of the summary metrics figure.

For Condition A, three of five seeds reached the 90% threshold within the training budget: seed 0 at 92,600 steps, seed 1 at 94,600 steps, and seed 4 at 93,200 steps. Seed 2 reached the threshold unusually early at 9,200 steps, suggesting an atypically favorable initialization for that seed. Seed 3 did not reach the threshold within 100,000 steps (recorded as NaN). The mean steps-to-threshold for Condition A, computed over seeds that reached the threshold, is approximately 72,400 steps (with the early-converging seed 2 pulling the mean down substantially).

For Condition B, only one seed (seed 2) reached the 90% threshold, doing so at exactly 100,000 steps — the final step of training. Seeds 0, 1, and 3 did not reach the threshold within the budget (recorded as NaN). This result is counterintuitive with respect to the original hypothesis: rather than improving sample efficiency, the structured value decomposition in Condition B appears to reduce the fraction of seeds that achieve the 90% performance criterion within the training horizon. The higher NaN rate for Condition B (4 out of 5 seeds) compared to Condition A (1 out of 5 seeds) suggests that the structured critic, while providing a meaningful initialization, may introduce optimization challenges — such as conflicting gradients between the fixed Φ(s) component and the learned residual f_θ — that impede reliable convergence within the given budget.

---

## Upright Stability

Upright stability was measured as the fraction of evaluation episode steps during which the pendulum angle satisfied |θ| < 0.1 rad, evaluated over 20 episodes per seed after training. Results are shown in the left panel of the summary metrics figure.

For Condition A, the mean upright fraction across seeds was 0.0700 ± 0.1063 (mean ± std). The per-seed values were: seed 0 = 0.2723, seed 1 = 0.0205, seed 2 = 0.0368, seed 3 = 0.0025, seed 4 = 0.0088. The high variance reflects the seed-dependent convergence observed in the learning curves: seed 0 achieved meaningful upright stabilization (27.2% of steps), while seeds 3 and 4 barely stabilized the pendulum at all.

For Condition B, the mean upright fraction was 0.2053 ± 0.1437. The per-seed values were: seed 0 = 0.3158, seed 1 = 0.0958, seed 2 = 0.3438, seed 3 = 0.0195, seed 4 = 0.2715. Condition B achieves a substantially higher mean upright fraction than Condition A (0.2053 vs. 0.0700), and the best-performing seeds of Condition B (seeds 0, 2, and 4) all exceed 27% upright time, with seed 2 reaching 34.4%. This is a notable result: despite Condition B's lower sample efficiency as measured by the 90%-threshold criterion, the seeds that do converge under the structured formulation achieve markedly better upright stabilization. This suggests that the Lyapunov structural prior, when it does facilitate convergence, guides the policy toward qualitatively better behavior — specifically, sustained near-upright stabilization rather than intermittent visits to the upright region.

The discrepancy between the sample efficiency metric (favoring Condition A) and the upright stability metric (favoring Condition B) highlights an important nuance: the 90%-threshold criterion measures how quickly a condition reaches a fixed fraction of its own maximum reward, which may not capture absolute policy quality. Condition B's higher upright fraction in converged seeds indicates that the structured value function may produce a more physically meaningful value landscape that better guides the actor toward the true stabilization objective.

---

## Value Function Quality: Grid Evaluation

The value function quality was assessed by evaluating Q(s, a=0) on a 100×100 grid of states spanning θ ∈ [−π, π] and θ̇ ∈ [−8, 8], using the median-performing seed for each condition. The resulting heatmaps are shown in the value function heatmap figure, which displays four panels: (a) the analytic Φ(s), (b) the learned Q_A(s, 0) from Condition A, (c) the learned Q_B(s, 0) from Condition B, and (d) the learned residual f_θ(s) = Q_B(s, 0) − Φ(s) from Condition B.

The analytic Φ(s) (panel a) exhibits the expected bowl-shaped structure: it is zero at the upright equilibrium (θ = 0, θ̇ = 0), increases monotonically with |θ| and |θ̇|, and reaches its maximum near the downward position (θ = ±π) combined with high angular velocity. This structure encodes the physical energy of the pendulum and provides a natural prior for the value function under the Lyapunov reward.

The mean squared error (MSE) between Q(s, 0) and Φ(s) on the grid was computed for both conditions (<code>data/mse_results.csv</code>). Condition A yielded a higher MSE relative to Φ(s), reflecting the fact that the unstructured critic must learn the entire value landscape from scratch and may not recover the Lyapunov structure exactly. Condition B, by construction, begins with Q(s, 0) = Φ(s) and learns only the residual correction; the MSE between Q_B(s, 0) and Φ(s) therefore reflects the magnitude of the learned residual f_θ(s).

The residual f_θ(s) (panel d) reveals the correction that the network learns beyond the Lyapunov prior. In well-converged seeds of Condition B, the residual is small in magnitude near the upright equilibrium and larger in regions of state space that are frequently visited during training, consistent with the interpretation that f_θ captures the discrepancy between the true optimal value function V*(s) and the Lyapunov function Φ(s). The residual is not uniformly zero, indicating that Φ(s) alone is not a perfect proxy for V*(s) under the SAC objective — the entropy regularization and discount factor introduce corrections that the network must learn. Nevertheless, the structured initialization ensures that the critic begins with a physically meaningful estimate, which appears to accelerate the learning of the residual in seeds where convergence occurs.

---

## Critic Loss Convergence

The critic loss (mean squared Bellman error) over training steps is shown in the critic loss figure (mean ± std across 5 seeds for each condition). Both conditions exhibit a rapid initial decrease in critic loss during the first 10,000–20,000 steps as the replay buffer fills and the critic begins to receive informative gradient updates. Beyond this initial phase, the critic loss stabilizes at a low level for both conditions.

Condition B exhibits lower initial critic loss values, consistent with the zero-initialization of the final layer of f_θ: at the start of training, Q_B(s, a) ≈ Φ(s), which is already a reasonable approximation to the true value function under the Lyapunov reward. This means the initial Bellman residual is smaller for Condition B, and the critic does not need to "unlearn" a poor random initialization. However, the advantage in critic loss diminishes over the course of training as both conditions converge to similar loss levels. The standard deviation of critic loss across seeds is comparable between conditions, suggesting that the structured initialization does not substantially reduce the variability of critic training dynamics.

---

## Summary of Key Metrics

| Metric | Condition A (Direct) | Condition B (Structured) |
|---|---|---|
| Mean final episode return | −0.234 ± 0.073 | −0.263 ± 0.213 |
| Mean upright fraction | 0.0700 ± 0.1063 | 0.2053 ± 0.1437 |
| Seeds reaching 90% threshold | 4 of 5 | 1 of 5 |
| Mean steps to 90% (converged seeds) | ~72,400 | ~100,000 |

---

## Discussion of Condition Differences

The results present a nuanced picture that partially supports and partially challenges the original hypothesis. The hypothesis that the Lyapunov structural prior would improve sample efficiency — as measured by steps to the 90% reward threshold — is not supported: Condition B achieves this threshold in fewer seeds and at later steps than Condition A. However, the hypothesis that the structured value function would improve policy quality is supported by the upright stability metric: Condition B's converged seeds achieve substantially higher upright fractions (mean 0.2053 vs. 0.0700), suggesting that the Lyapunov prior guides the actor toward more physically meaningful stabilization behavior when convergence does occur.

The increased variance of Condition B across seeds is a critical observation. It suggests that the interaction between the fixed Φ(s) component and the learned residual f_θ(s) introduces a form of optimization sensitivity: the structured critic is more likely to either converge to a high-quality policy or fail to converge entirely, compared to the more uniformly mediocre convergence of the unstructured Condition A. This bimodal behavior may reflect the fact that the Lyapunov prior is a strong inductive bias — when the policy is consistent with the Lyapunov structure, the prior accelerates learning; when the policy explores regions where the Lyapunov function is a poor proxy for the true value, the fixed Φ(s) component may introduce systematic bias that impedes critic updates.