![RLForge Logo](docs/source/_static/logo.svg)

# RLForge

![docs](https://readthedocs.org/projects/rlforge/badge/?version=latest)
![PyPI - License](https://img.shields.io/pypi/l/rlforge)
![PyPI - Version](https://img.shields.io/pypi/v/rlforge)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rlforge)
![PyPI Downloads](https://pepy.tech/badge/rlforge)

**RLForge** is an open-source reinforcement learning library that makes it easy to
experiment with RL algorithms, environments, and training workflows. It is designed
to be lightweight, educational, and fully compatible with the
[Gymnasium](https://gymnasium.farama.org/) ecosystem (formerly OpenAI Gym).

![Lunar Lander DQN](docs/source/_static/lunarLander.gif)

## Features

- **Educational algorithms:** from simple **multi-armed bandits** and **tabular methods**
  (SARSA, Q-learning, Expected SARSA) to function approximation with linear models and MLPs.
- **Advanced deep RL agents:** including **DQN**, **REINFORCE**, **Actor-Critic**, **DDPG**,
  **TD3**, **SAC**, and **PPO** (both discrete and continuous).
- **Custom environments** — bandits, short corridor, maze variations, robotics-inspired tasks
  like Mecanum Car, and classic control problems such as Pendulum.
- **Gymnasium compatibility:** seamlessly integrate RLForge agents with hundreds of
  standardized benchmark environments.
- **Visualization tools:** built-in experiment runner and plotting utilities for learning
  curves, episode statistics, and trajectory tracking.
- **PyTorch integration:** optional install enables neural-network-based agents:
  - `DQNTorchAgent`
  - `DDPGAgent`
  - `TD3Agent`
  - `SACAgent`
  - `PPODiscrete`
  - `PPOContinuous`
  
  These PyTorch agents also support **vectorized environments**, allowing parallel training
  across multiple instances for faster and more stable learning.

## Installation

If you already have Python installed, you can install RLForge with:

```console
pip install rlforge
```

This will download and install the latest stable release of `rlforge` available in the
[Python Package Index](https://pypi.org/project/rlforge/).

RLForge works with **Python 3.11 or later**. Installing with `pip` will automatically
download all required dependencies if they are not already present.

### Optional PyTorch Support

To enable PyTorch-based agents, install RLForge with the `torch` extra:

```console
pip install rlforge[torch]
```

Or install all optional dependencies:

```console
pip install rlforge[all]
```

## Documentation

Full documentation, including tutorials and examples, is available on [Read the Docs](https://rlforge.readthedocs.io).

Explore the examples section to see RLForge in action, from simple bandit problems
to advanced continuous control tasks.
