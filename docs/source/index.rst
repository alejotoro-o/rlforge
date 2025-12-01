.. image:: _static/logo.svg
   :alt: RLForge Logo
   :align: center
   :height: 150px

Welcome to RLForge's documentation!
===================================

**RLForge** is a lightweight yet powerful reinforcement learning framework designed
to make RL accessible to everyone, from students exploring bandits to researchers
experimenting with advanced deep RL algorithms.

.. image:: _static/lunarLander.gif
   :alt: Lunar Lander Environment Demo
   :align: center
   :width: 500px

What is RLForge?
----------------
RLForge provides a unified interface for building, training, and evaluating
reinforcement learning agents across a wide variety of environments. It is
designed to be:

- **Educational:** clear implementations of classic algorithms like
  multi-armed bandits, SARSA, and Q-learning.
- **Scalable:** support for advanced deep RL methods such as DQN, PPO,
  TD3, SAC, and DDPG.
- **Compatible:** works seamlessly with `Gymnasium <https://gymnasium.farama.org/>`_
  environments, plus custom environments included in RLForge (mazes, bandits,
  short corridor, robotics-inspired tasks, etc.).
- **Visual:** built-in experiment runner and plotting utilities for
  analyzing learning curves and trajectories.

Algorithms
------------------
RLForge includes a wide spectrum of RL agents:

- **Basic algorithms:** Bandits, Tabular SARSA, Q-learning, Expected SARSA.
- **Function approximation:** Linear regression, MLP-based agents.
- **Deep RL (PyTorch-based):**
  - DQNTorchAgent
  - DDPGAgent
  - TD3Agent
  - SACAgent
  - PPODiscrete
  - PPOContinuous

PyTorch Agents and Vectorized Environments
------------------------------------------
Agents implemented with PyTorch not only leverage neural networks for
function approximation, but also support **vectorized environments**.
This allows training across multiple parallel environments, dramatically
improving sample efficiency and stability.

Getting Started
---------------
- Install RLForge with:

  .. code-block:: console

     pip install rlforge

- For PyTorch-based agents, install with the optional dependency:

  .. code-block:: console

     pip install rlforge[torch]

Explore the examples section to see RLForge in action, from simple bandit
problems to advanced continuous control tasks.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   environments
   examples

.. toctree::
   :maxdepth: 1
   :caption: Agents

   agents/base_agent
   agents/bandit_agent
   tabular_agents
   semi_gradient_agents
   policy_gradient_agents

.. toctree::
   :maxdepth: 1
   :caption: Experiment Runner

   experiment_runner

.. toctree::
   :maxdepth: 1
   :caption: Policies

   policies

.. toctree::
   :maxdepth: 1
   :caption: Utils

   utils


