Installation
============

If you already have Python installed on your computer, you can install RLForge with:

.. code-block:: console

    pip install rlforge

This will download and install the latest stable release of ``rlforge`` available in the
`Python Package Index <https://pypi.org/>`_.

Requirements
------------
RLForge works with **Python 3.9 or later** (the package metadata specifies Python >=3.10)
and depends on:

- `NumPy <https://numpy.org/>`_ (numerical computations)
- `tqdm <https://tqdm.github.io/>`_ (progress bars)
- `Gymnasium <https://gymnasium.farama.org/>`_ (environments)
- `Matplotlib <https://matplotlib.org/>`_ (plotting)

Installing with ``pip`` will automatically download these dependencies if they are not
already present in your workspace.

Optional Dependencies
---------------------
RLForge also provides optional support for **PyTorch**. If you want to run agents that
use deep neural networks implemented in PyTorch, install RLForge with the ``torch`` extra:

.. code-block:: console

    pip install rlforge[torch]

or install all optional dependencies with:

.. code-block:: console

    pip install rlforge[all]

This will install:

- `torch <https://pytorch.org/>`_ (>=2.9.0)

PyTorch is required for the following agents:

- **DQNTorchAgent** — Deep Q-Networks implemented in PyTorch.
- **DDPGAgent** — Deep Deterministic Policy Gradient.
- **TD3Agent** — Twin Delayed Deep Deterministic Policy Gradient.
- **SACAgent** — Soft Actor-Critic.
- **PPODiscrete** — Proximal Policy Optimization for discrete action spaces.
- **PPOContinuous** — Proximal Policy Optimization for continuous action spaces.

If you only plan to use tabular or NumPy-based agents, you can install RLForge without
the ``torch`` extra.

Package Metadata
----------------
For reference, RLForge’s ``pyproject.toml`` specifies:

.. code-block:: toml

    [build-system]
    requires = ["setuptools>=61.0"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "rlforge"
    version = "1.0.0"
    authors = [
        { name="Alejandro Toro-Ossaba", email="alejotoro.o@gmail.com" },
    ]
    description = "Reinforcement Learning for Everyone."
    readme = "README.md"
    requires-python = ">=3.10"
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
    keywords = ["reinforcement learning", "machine learning"]
    license = {file = "LICENSE"}
    dependencies = [
        "numpy>=2.3.4",
        "tqdm>=4.67.1",
        "gymnasium>=1.2.2",
        "matplotlib>=3.10.7",
    ]
    [project.optional-dependencies]
    torch = [
        "torch>=2.9.0",
    ]
    all = [
        "torch>=2.9.0",
    ]

    [project.urls]
    "Homepage" = "https://github.com/alejotoro-o/rlforge"
    "Documentation" = "https://rlforge.readthedocs.io/"
    "Repository" = "https://github.com/alejotoro-o/rlforge"
    "Bug Tracker" = "https://github.com/alejotoro-o/rlforge/issues"