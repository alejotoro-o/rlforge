Tabular Agents
==============

Tabular agents implement reinforcement learning algorithms that store and update
action-value estimates in explicit tables. These methods are well-suited for
small, discrete state and action spaces, where values can be represented directly
without the need for function approximation. They provide a clear and educational
foundation for understanding how agents learn from experience.

RLForge includes the following tabular agents:

- **SARSA Agent** — learns action values using the on-policy SARSA algorithm,
  updating estimates based on the action actually taken in the next state.
- **Q Agent** — implements the classic Q-learning algorithm, an off-policy method
  that updates values toward the greedy action in the next state.
- **Expected SARSA Agent** — a variant of SARSA that uses the expected value of
  all possible actions in the next state, weighted by the policy's probabilities,
  leading to smoother updates.

These agents are ideal for experimenting with fundamental reinforcement learning
concepts and serve as building blocks for more advanced methods.

.. toctree::
    :maxdepth: 1

    agents/tabular/sarsa_agent
    agents/tabular/q_agent
    agents/tabular/expected_sarsa_agent