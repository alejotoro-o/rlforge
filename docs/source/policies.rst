Policies
========

Policies define how an agent selects actions given its current knowledge or
preferences. They are the core mechanism that balances **exploration** (trying
new actions to discover rewards) and **exploitation** (choosing the best-known
action to maximize reward). RLForge provides several standard policies that
illustrate different approaches to this trade-off:

- **Epsilon-Greedy** — selects the best-known action most of the time, but with
  a small probability `epsilon` chooses a random action to encourage exploration.
- **Softmax** — assigns probabilities to actions based on their estimated values,
  controlled by a `temperature` parameter that adjusts the balance between
  greediness and randomness.
- **Gaussian** — samples continuous actions from a normal distribution defined
  by a mean (`mu`) and standard deviation (`sigma`), useful for environments
  with continuous action spaces.

These policies serve as building blocks for agents in RLForge, allowing you to
experiment with different exploration strategies and adapt them to discrete or
continuous environments.


.. toctree::
    :maxdepth: 1

    policies/epsilon_greedy
    policies/softmax
    policies/gaussian