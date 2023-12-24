# RLForge

![docs](https://readthedocs.org/projects/rlforge/badge/?version=latest)
![PyPI - License](https://img.shields.io/pypi/l/rlforge)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rlforge)
![PyPI Downloads](https://pepy.tech/badge/rlforge)

RL Forge is an open source reinforcement learning library that aims to provide the users with useful functions for the development of Reinforcement Learning Agents. The library also includes multiple popular reinforcement learning agents and environments, in addition, it is designed to be compatible with the gymnasium library (previous OpenAI Gym).

## Installation

If you already have Python installed in your computer, you can install RLForge with:

    pip install rlforge

This will download and install the latest stable release of ``rlforge`` available in the [Python Package Index](https://pypi.org/project/rlforge/).

RLForge works with Python 3.9 or later, intalling RLForge with ``pip`` will automatically download all required packages if they are not present in your workspace.

## Documentation

The documentation, with examples, can be found in [Read the Docs](https://rlforge.readthedocs.io) (**NOTE:** Currently the documentation is under development and is not totaly complete).

## Examples

Multiple examples on how to use the different agents are included in the [examples folder](https://github.com/alejotoro-o/rlforge/tree/main/examples). These examples include using the library both with gymnasium environments and environments included in this package.

- [SARSA - Frozen Lake](https://github.com/alejotoro-o/rlforge/blob/main/examples/sarsa_frozenLake.ipynb)
- [Dyna Architecture - Planning Agents](https://github.com/alejotoro-o/rlforge/blob/main/examples/dynaArchitecture_planningAgents.ipynb)
- [Tabular Methods Comparison](https://github.com/alejotoro-o/rlforge/blob/main/examples/tabularMethods_comparison.ipynb)
- [Function Approximation with Tile Coding and Q learning - Mountain Car](https://github.com/alejotoro-o/rlforge/blob/main/examples/linearFunctionApproximation_mountainCar.ipynb)
- [Tile Coding Q learning - Mecanum Car Environment](https://github.com/alejotoro-o/rlforge/blob/main/examples/qlearning_mecanumCar.ipynb)
- [Tile Coding Q learning - Obstacle Avoidance Environment](https://github.com/alejotoro-o/rlforge/blob/main/examples/obstacle_avoidance.ipynb)
- [Tile Coding Q learning - Trajectory Tracking Environment](https://github.com/alejotoro-o/rlforge/blob/main/examples/trajectory_tracking.ipynb)
- [DQN - Mountain Car](https://github.com/alejotoro-o/rlforge/blob/main/examples/DQN_mountainCar.ipynb)
- [Softmax and Gaussian Actor Critic - Pendulum](https://github.com/alejotoro-o/rlforge/blob/main/examples/actorCritic_pendulum.ipynb)