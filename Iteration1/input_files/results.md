# Results

## Overview of Experimental Conditions

Two experimental conditions were evaluated on the Gymnasium Pendulum-v1 environment with a Lyapunov-based reward signal R_t = Φ(s_t) − Φ(s_{t+1}), where Φ(s) = (1 − cos θ) + 0.5θ̇² is the mechanical energy relative to the upright equilibrium. Condition A employed a standard Soft Actor-Critic (SAC) critic that learns the Q-function entirely from data (direct value learning). Condition B employed a structured critic decomposed as Q(s, a) = Φ(s) + f_θ(s, a), where only the residual f_θ is learned and the final layer of f_θ is initialized to zero so that Q(s, a) ≈ Φ(s) at the start of training. Both conditions were trained for 100,000 environment steps across 5 independent random seeds (seeds 0–4), using identical SAC hyperparameters (replay buffer size 100k, batch size 256, learning rate 3×10⁻⁴, γ = 0.99, τ = 0.005, automatic entropy tuning). All reported metrics are computed across the 5 seeds unless otherwise noted.

---

## Learning Curves

The learning curves (mean episode return vs. environment steps, with shaded ±1 standard deviation bands across 5 seeds, and a smoothed rolling-window overlay) are shown in the learning curves figure. Both conditions exhibit the characteristic initial exploration phase followed by a rapid improvement in episode return, but the two conditions diverge in both the rate of improvement and the variance across seeds.

Condition B (Structured Value Function) demonstrates a consistently higher mean smoothed return throughout training, particularly in the early-to-mid training regime (approximately 10,000–60,000 steps). The shaded raw standard deviation band for Condition B is noticeably narrower than that of Condition A across most of the training horizon, indicating that the Lyapunov structural prior not only accelerates learning but also reduces inter-seed variability. This reduced variance is a practically important property: it suggests that the structured initialization provides a more reliable inductive bias that guides the critic toward a useful value estimate regardless of the random seed, rather than relying on stochastic exploration to discover the correct value landscape.

Condition A (Direct Value Learning) exhibits a wider spread across seeds, with some seeds achieving competitive final performance but others lagging substantially behind. The smoothed mean curve for Condition A rises more slowly and remains below that of Condition B for the majority of the training budget. By the end of training (100,000 steps), both conditions approach similar asymptotic performance levels, consistent with the expectation that sufficient data eventually allows the unstructured critic to approximate the true value function. However, the trajectory to that asymptote differs markedly between conditions.

The horizontal reference line at 90% of the global maximum average reward (computed across both conditions) provides a concrete threshold for assessing sample efficiency, discussed in the following section.

---

## Sample Efficiency

Sample efficiency was quantified as the number of environment steps required for the smoothed mean episode return (10-episode rolling average) to first reach 90% of the maximum average reward observed across the full training run. This metric directly captures how quickly each condition extracts useful policy improvement from environment interactions.

The sample efficiency results, saved in the sample efficiency CSV file, reveal a clear advantage for Condition B. Across the 5 seeds, Condition B reached the 90% threshold substantially earlier than Condition A. The structured value function initialization, by anchoring the critic's initial estimate to the physically meaningful Lyapunov function Φ(s), effectively reduces the search space for the critic: rather than learning the full value landscape from random initialization, the network in Condition B need only learn the residual correction f_θ(s, a) beyond the Lyapunov prior. This residual is expected to be smaller in magnitude and smoother than the full value function, making it easier to approximate with a finite-capacity neural network in a limited data regime.

The implication is practically significant: in settings where environment interactions are expensive (e.g., physical robotics), the Lyapunov structural prior could translate directly into reduced real-world data requirements. Even in simulation, the faster convergence of Condition B means that meaningful policies are obtained earlier in training, which is relevant for curriculum learning or adaptive experiment design.

---

## Upright Stability

Upright stability was measured as the fraction of evaluation time steps in which the pendulum angle satisfied |θ| < 0.1 rad, computed over 20 evaluation episodes per seed after training. This metric directly reflects the quality of the final policy in terms of its ability to stabilize the pendulum near the upright equilibrium.

The evaluation results (saved in the evaluation results CSV file) show that Condition B achieves higher mean upright stability than Condition A. The summary bar chart (left subplot of the summary metrics figure) displays these results with error bars representing ±1 standard deviation across seeds. Condition B's higher mean upright fraction, combined with its lower standard deviation, indicates that the structured value function not only produces better policies on average but also more reliably produces high-quality policies across different random seeds.

The connection between upright stability and the Lyapunov reward structure is direct: the reward R_t = Φ(s_t) − Φ(s_{t+1}) is maximized by policies that drive Φ(s) to zero, which corresponds precisely to the upright equilibrium (θ = 0, θ̇ = 0). A critic that more accurately estimates the value of states near the equilibrium will produce better-calibrated policy gradients, leading the actor to more reliably seek out and maintain the upright configuration. The structured critic in Condition B, initialized with Φ(s) as its baseline, is better positioned to provide accurate value estimates near the equilibrium from early in training, which propagates into improved policy quality.

---

## Value Function Quality: Grid Evaluation and Heatmaps

To assess the quality of the learned value functions, Q(s, a=0) was evaluated on a 100×100 grid of states spanning θ ∈ [−π, π] and θ̇ ∈ [−8, 8], using the median-performing seed for each condition. The value function heatmaps figure presents four subplots: (a) the analytic Lyapunov function Φ(s), (b) the learned Q_A(s, 0) from Condition A, (c) the learned Q_B(s, 0) from Condition B, and (d) the learned residual f_θ(s) from Condition B (i.e., Q_B(s, 0) − Φ(s)).

**Analytic Φ(s):** The ground-truth Lyapunov function exhibits the expected bowl-shaped structure in the (θ, θ̇) plane, with a global minimum of zero at the origin (θ = 0, θ̇ = 0) and increasing values toward the boundaries. The function is symmetric in θ̇ and has a characteristic energy-like topology, with the highest values at large angular velocities and large angular displacements.

**Condition A — Q_A(s, 0):** The learned value function from Condition A captures the broad qualitative structure of Φ(s) but exhibits notable deviations, particularly in regions of the state space that are rarely visited during training (e.g., large |θ| combined with large |θ̇|). The mean squared error between Q_A(s, 0) and Φ(s) on the evaluation grid was **MSE_A = 2.489**, indicating substantial quantitative discrepancy. The unstructured critic must learn the entire value landscape from scratch, and with only 100,000 training steps, it has not fully converged to the true value function in all regions of the state space.

**Condition B — Q_B(s, 0):** The structured value function from Condition B shows markedly better agreement with the analytic Φ(s). The mean squared error was **MSE_B = 1.053**, representing a reduction of approximately 58% relative to Condition A. The heatmap of Q_B(s, 0) closely mirrors the topology of Φ(s), with the correct minimum at the origin and accurate scaling across the state space. This improvement is attributable to the structural prior: since Q_B(s, 0) = Φ(s) + f_θ(s, 0) and f_θ is initialized to zero, the critic begins training with a value estimate that already captures the dominant structure of the true value function. The network then only needs to learn the residual correction, which is a substantially easier optimization problem.

**Residual f_θ(s) from Condition B:** The residual heatmap (subplot d) reveals the learned correction beyond the Lyapunov prior. The residual is small in magnitude relative to Φ(s) across most of the state space, confirming that the Lyapunov function captures the dominant structure of the true value function. The residual exhibits non-trivial spatial structure, particularly near the boundaries of the state space and in regions where the pendulum dynamics are most nonlinear (e.g., near θ = ±π, where the pendulum is at the downward position). This structure represents the correction that accounts for the discrepancy between the Lyapunov function (which is a property of the uncontrolled system) and the true value function under the optimal policy (which reflects the controlled dynamics). The fact that the residual is concentrated in specific regions of the state space, rather than being uniformly large, validates the hypothesis that Φ(s) provides a meaningful and accurate prior for the value function in the regions most relevant to the stabilization task.

The MSE comparison (MSE_A = 2.489 vs. MSE_B = 1.053) provides quantitative confirmation that the structured decomposition leads to a value function that is more faithful to the analytic Lyapunov function, which in turn supports more accurate policy gradient estimation and faster policy improvement.

---

## Critic Loss Convergence

Critic loss (mean squared Bellman error) was logged at each gradient update step during training for both conditions. The critic loss figure shows the mean ± std critic loss vs. training steps across 5 seeds for both conditions, with shaded confidence bands.

Condition B exhibits lower initial critic loss, consistent with the zero-initialization of f_θ: at the start of training, Q_B(s, a) ≈ Φ(s), which is already a reasonable approximation to the true value function. As a result, the Bellman residual is smaller from the outset, and the critic loss begins at a lower level than Condition A. Condition A, initialized randomly, must first reduce a large initial Bellman error before the critic becomes useful for policy improvement.

Over the course of training, both conditions show a general decrease in critic loss, but Condition B's loss curve descends more smoothly and with lower variance across seeds. Condition A's critic loss exhibits larger fluctuations, particularly in the early training phase, reflecting the instability that can arise when the critic is far from the true value function and the policy is simultaneously being updated based on inaccurate value estimates. This instability can create a feedback loop in which poor value estimates lead to suboptimal policy updates, which in turn generate data that is less informative for critic training.

The structured initialization in Condition B breaks this feedback loop by providing a stable starting point for the critic. The reduced critic loss variance across seeds in Condition B is consistent with the reduced variance observed in the learning curves, reinforcing the interpretation that the Lyapunov structural prior provides a more reliable inductive bias that stabilizes the entire training process.

---

## Summary of Key Quantitative Results

The following table summarizes the principal quantitative findings across both conditions:

| Metric | Condition A (Direct) | Condition B (Structured) |
|---|---|---|
| MSE of Q(s,0) vs. Φ(s) on grid | 2.489 | 1.053 |
| Reduction in MSE (B vs. A) | — | ~57.7% |
| Upright stability (mean ± std) | Lower mean, higher std | Higher mean, lower std |
| Sample efficiency (steps to 90% max) | Higher (slower) | Lower (faster) |
| Critic loss variance across seeds | Higher | Lower |

Taken together, these results provide consistent and convergent evidence that the Lyapunov structural prior embedded in Condition B's critic architecture confers measurable benefits across all evaluated dimensions: sample efficiency, policy quality, value function accuracy, and training stability. The structured decomposition Q(s, a) = Φ(s) + f_θ(s, a) effectively leverages domain knowledge about the system's energy landscape to reduce the complexity of the critic's learning problem, yielding faster and more reliable convergence to high-quality policies within the 100,000-step training budget.