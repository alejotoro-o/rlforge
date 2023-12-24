# RLForge

![docs](https://readthedocs.org/projects/rlforge/badge/?version=latest)
![PyPI - License](https://img.shields.io/pypi/l/rlforge)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rlforge)
![PyPI Downloads](https://pepy.tech/badge/rlforge)

RL Forge is an open source reinforcement learning library that aims to provide the users with useful functions for the development of Reinforcement Learning Agents. The library also includes multiple popular reinforcement learning agents and environments, in addition, it is designed to be compatible with the gymnasium library (previous OpenAI Gym).

## Installation

If you already have Python installed in your computer, you can install RLForge with:

    pip install rlforge

This will download and install the latest stable release of ``rlforge`` available in the `Python Package Index <https://pypi.org/>`_.

RLForge works with Python 3.9 or later, and depends on `NumPy <https://numpy.org/>`_. Intalling with ``pip`` will automatically download it if it's not present in your workspace.

## Documentation

The documentation, with examples, can be found in [Read the Docs](https://rlforge.readthedocs.io) (**NOTE:** Currently the documentation is under development and is not totaly complete).

## Examples

Multiple examples on how to use the different agents are included in the [examples folder](examples). These examples include using the library both with gymnasium environments and environments included in this package.

- [SARSA - Frozen Lake](examples/sarsa_frozenLake.ipynb)
- [Dyna Architecture - Planning Agents](examples/dynaArchitecture_planningAgents.ipynb)
- [Tabular Methods Comparison](examples/tabularMethods_comparison.ipynb)
- [Function Approximation with Tile Coding and Q learning - Mountain Car](examples/linearFunctionApproximation_mountainCar.ipynb)
- [Tile Coding Q learning - Mecanum Car Environment](examples/qlearning_mecanumCar.ipynb)
- [Tile Coding Q learning - Obstacle Avoidance Environment](examples/obstacle_avoidance.ipynb)
- [Tile Coding Q learning - Trajectory Tracking Environment](examples/trajectory_tracking.ipynb)
- [DQN - Mountain Car](examples/DQN_mountainCar.ipynb)
- [Softmax and Gaussian Actor Critic - Pendulum](examples/actorCritic_pendulum.ipynb)