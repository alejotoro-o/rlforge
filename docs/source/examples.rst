Examples
========

The following examples illustrate how to apply RLForge agents and environments
to a variety of reinforcement learning problems. They range from simple
tabular methods to advanced deep reinforcement learning algorithms, and
cover both discrete and continuous action spaces. Each example is designed
to highlight a specific concept, algorithm, or environment.

- **K-Bandits** — demonstrates exploration strategies and action selection
  in multi-armed bandit problems.
- **SARSA on FrozenLake** — shows on-policy temporal-difference learning
  in a discrete gridworld environment with stochastic transitions.
- **Q-Learning on MecanumCar** — applies off-policy Q-learning to a robotics-inspired
  continuous navigation task.
- **Tabular Methods Comparison** — compares SARSA, Q-learning, and Expected SARSA
  side by side in a shared environment.
- **Dyna Architecture** — illustrates model-based reinforcement learning with
  planning updates using the Dyna-Q algorithm.
- **Linear Function Approximation** — demonstrates generalization using linear
  approximators instead of tabular state representations.
- **DQN on MountainCar** — applies Deep Q-Networks to the classic continuous-state
  MountainCar environment.
- **DQN (PyTorch) on CartPole** — shows a PyTorch implementation of DQN for balancing
  the CartPole environment.
- **REINFORCE on Short Corridor** — implements the Monte Carlo policy gradient
  algorithm in Sutton & Barto’s Short Corridor example.
- **Actor-Critic on Pendulum** — demonstrates the actor-critic architecture in a
  continuous control task.
- **DDPG on Pendulum** — applies Deep Deterministic Policy Gradient to the pendulum
  swing-up problem.
- **TD3 on Pendulum** — shows improvements over DDPG using twin critics and delayed
  updates.
- **SAC on Pendulum** — implements Soft Actor-Critic, maximizing both reward and
  entropy for robust exploration.
- **PPO (Discrete) on CartPole** — applies Proximal Policy Optimization to a discrete
  control task.
- **PPO (Continuous) on Pendulum** — applies PPO with Gaussian policies to a continuous
  control task.

.. toctree::
   :maxdepth: 1

   examples/k_bandits
   examples/sarsa_frozenLake
   examples/qlearning_mecanumCar
   examples/tabularMethods_comparison
   examples/dynaArchitecture
   examples/linearFunctionApproximation
   examples/dqn_mountainCar
   examples/dqn_pytorch_cartPole
   examples/reinforce_shortCorridor
   examples/actorCritic_pendulum
   examples/ddpg_pendulum
   examples/td3_pendulum
   examples/sac_pendulum
   examples/ppo_cartPole
   examples/ppo_pendulum