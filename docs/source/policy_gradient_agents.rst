Policy Gradient Agents
======================

Policy gradient methods directly optimize the agent's policy by adjusting its
parameters in the direction that increases expected rewards. Unlike value-based
methods, which learn action-value functions and derive policies indirectly,
policy gradient agents learn **stochastic policies** that can naturally handle
both discrete and continuous action spaces.

A common approach is the **actor-critic architecture**, where:

- The **actor** represents the policy and selects actions.
- The **critic** estimates value functions and provides feedback to improve
  the actor's parameters.

RLForge currently includes:

- **REINFORCE** — the classic Monte Carlo policy gradient algorithm that updates
  parameters directly based on returns, without a critic.
- **Softmax Actor-Critic** — uses a softmax distribution over discrete actions,
  allowing the agent to balance exploration and exploitation while learning
  directly from policy gradients.
- **Gaussian Actor-Critic** — outputs continuous actions by sampling from a
  Gaussian distribution parameterized by mean and variance. The current
  implementation supports a single continuous output, making it suitable for
  environments with one-dimensional action spaces.
- **Deep Deterministic Policy Gradient (DDPG)** — an off-policy actor-critic
  method for continuous control, using deterministic policies and target networks.
- **Twin Delayed Deep Deterministic Policy Gradient (TD3)** — improves DDPG by
  using twin critics, delayed policy updates, and target policy smoothing.
- **Soft Actor-Critic (SAC)** — an off-policy actor-critic method that maximizes
  both reward and entropy, encouraging exploration with stochastic policies.
- **Proximal Policy Optimization (PPO-Discrete)** — applies PPO to discrete
  action spaces, using clipped objectives and GAE for stable updates.
- **Proximal Policy Optimization (PPO-Continuous)** — applies PPO to continuous
  action spaces, supporting Gaussian policies with optional tanh squashing.
  
.. toctree::
    :maxdepth: 1

    agents/policy_gradient/reinforce
    agents/policy_gradient/softmax_actor_critic
    agents/policy_gradient/gaussian_actor_critic
    agents/policy_gradient/ddpg
    agents/policy_gradient/td3
    agents/policy_gradient/sac
    agents/policy_gradient/ppo_discrete
    agents/policy_gradient/ppo_continuous