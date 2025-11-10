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

- **Softmax Actor-Critic** — uses a softmax distribution over discrete actions,
  allowing the agent to balance exploration and exploitation while learning
  directly from policy gradients.
- **Gaussian Actor-Critic** — outputs continuous actions by sampling from a
  Gaussian distribution parameterized by mean and variance. The current
  implementation supports a single continuous output, making it suitable for
  environments with one-dimensional action spaces.

  
.. toctree::
    :maxdepth: 1

    agents/policy_gradient/softmax_actor_critic
    agents/policy_gradient/gaussian_actor_critic