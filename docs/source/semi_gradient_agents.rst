Semi-gradient Agents
====================

Semi-gradient agents extend reinforcement learning beyond tabular methods by
using **function approximation** to estimate value functions. Instead of storing
explicit tables for every state–action pair, these agents generalize across
states, making them suitable for larger or continuous state spaces.

The term *semi-gradient* refers to the fact that updates are performed using
gradient descent on the value function approximation, but the target itself
depends on the parameters being updated. This approach is widely used in
practical reinforcement learning.

RLForge currently includes:

- **Linear Semi-gradient Agent** — uses linear function approximation with
  **tile coding** to represent features. Tile coding enables efficient
  generalization across continuous state spaces while maintaining interpretability.
- **Deep Q-Network (DQN)** — leverages deep neural networks to approximate
  action-value functions, enabling agents to handle high-dimensional inputs
  such as images or complex environments.

These agents demonstrate how function approximation techniques can scale
reinforcement learning beyond simple tabular domains, forming the foundation
for modern RL applications.

.. toctree::
    :maxdepth: 1

    agents/semi_gradient/linear_sg_agent
    agents/semi_gradient/dqn