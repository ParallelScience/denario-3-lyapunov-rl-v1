**Title: Lyapunov-Structured Value Functions for Reinforcement Learning on the Pendulum**

This research investigates whether structuring the critic as $V(s) = \Phi(s) + f_\theta(s)$ — where $\Phi(s)$ is an analytically given Lyapunov function and only the residual $f_\theta(s)$ is learned — improves sample efficiency, stability, and policy quality compared to learning $V(s)$ directly from data.

The environment is Gymnasium Pendulum-v1, a standard continuous-control benchmark where the goal is to stabilize the pendulum at the upright position ($\theta = 0$). The native environment reward is replaced entirely by the Lyapunov-based reward:

$$R_t = \Phi(s_t) - \Phi(s_{t+1})$$

where the Lyapunov function is the mechanical energy of the pendulum relative to the upright equilibrium:

$$\Phi(s) = (1 - \cos\theta) + \frac{1}{2}\dot\theta^2$$

This reward is positive when the agent drives the system toward the equilibrium ($\Phi \to 0$) and zero at the goal.

Two experimental conditions are compared using Soft Actor-Critic (SAC):

**Condition A — Direct Value Learning:** Standard SAC where the critic learns $V(s)$ entirely from data. This is the baseline.

**Condition B — Structured Value Function:** The critic is decomposed as $V(s) = \Phi(s) + f_\theta(s)$. The network outputs only the residual $f_\theta(s)$; the analytic $\Phi(s)$ is added directly. The hypothesis is that initializing the value estimate with the physically meaningful Lyapunov function reduces the residual's search space, leading to faster convergence and a more stable critic.

Both conditions are trained for 100,000 environment steps across 5 random seeds. Performance is evaluated by: (1) learning curves of the Lyapunov reward over time, (2) sample efficiency (steps to reach 90% of maximum reward), (3) upright stabilization time after training, and (4) a comparison of the learned value function against the analytic $\Phi(s)$ on a grid of states to assess how much structure the residual captures.
