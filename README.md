# RLForge

![docs](https://readthedocs.org/projects/rlforge/badge/?version=latest)
![PyPI - License](https://img.shields.io/pypi/l/rlforge)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rlforge)
![PyPI Downloads](https://pepy.tech/badge/rlforge)

RL Forge is an open source reinforcement learning library that aims to provide the users with useful functions for the development of Reinforcement Learning Agents. The library also includes multiple popular reinforcement learning agents and environments, in addition, it is designed to be compatible with the gymnasium library (previous OpenAI Gym).

## Installation

If you already have Python installed in your computer, you can install RLForge with:

.. code-block:: console

    pip install rlforge

This will download and install the latest stable release of ``rlforge`` available in the `Python Package Index <https://pypi.org/>`_.

RLForge works with Python 3.9 or later, and depends on `NumPy <https://numpy.org/>`_. Intalling with ``pip`` will automatically download it if it's not present in your workspace.

## Documentation

The documentation, with examples, can be found in [Read the Docs](https://rlforge.readthedocs.io) (**NOTE:** Currently the documentation is under development and is not totaly complete).

## Examples

- [SARSA Frozen Lake](examples/sarsa_forzenLake.ipynb)
- [Tabular Methods Comparison](examples/tabularMethods_comparison.ipynb)
- [DQN Mountain Car](examples/DQN_mountainCar.ipynb)