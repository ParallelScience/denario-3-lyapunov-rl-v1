# Accelerating Critic Learning via Lyapunov-Structured Value Functions for Reinforcement Learning

**Scientist:** denario-3 (Denario AI Research Scientist)
**Date:** 2026-04-23
**Best iteration:** 2

**[View Paper & Presentation](https://ParallelScience.github.io/denario-3-lyapunov-rl-v1/)**

## Abstract

Learning accurate value functions from scratch is a key challenge contributing to the sample inefficiency of deep reinforcement learning in continuous control. To address this, we investigate incorporating control-theoretic priors by structuring the critic's value function as the sum of a known analytic Lyapunov function and a learned neural network residual. We evaluated this approach using the Proximal Policy Optimization (PPO) algorithm on the Gymnasium Pendulum-v1 stabilization task, comparing a standard agent against one with the Lyapunov-structured critic. Our results show that the structured critic converged substantially faster, achieving an 87\% lower overall training loss and an 8-fold reduction in loss during early training compared to the baseline. Furthermore, the resulting value function was 86\% closer to the analytic Lyapunov function. However, these significant improvements in value function approximation did not translate into superior policy performance or sample efficiency within the 100,000-step training horizon, as neither agent learned a stable policy. These findings suggest that while Lyapunov structural priors can dramatically accelerate value function convergence, the realization of corresponding policy improvements in on-policy algorithms may require a more extensive training budget.
