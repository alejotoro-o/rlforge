Environments
============

The **RLForge environments** provide a diverse set of testbeds for experimenting
with reinforcement learning agents. They are designed to illustrate different
challenges such as exploration, control, and navigation, while remaining lightweight
and easy to integrate with your agents.  

In addition to these built-in environments, RLForge is fully compatible with all
`Gymnasium <https://gymnasium.farama.org/index.html>`_ environments, allowing you
to extend experiments to a wide variety of standardized benchmarks.

Available environments
-----------------------

- **Dyna Maze** — a grid-based maze environment ideal for testing planning and
  model-based reinforcement learning algorithms.
- **Shortcut Maze** — a variation of the maze environment with multiple paths,
  emphasizing exploration and the discovery of efficient trajectories.
- **Pendulum** — a classic continuous control task where the agent learns to
  balance and swing up a pendulum.
- **Mecanum Car** — a robotics-inspired environment simulating a mecanum-wheeled
  vehicle, useful for testing control in multi-dimensional continuous spaces.
- **Obstacle Avoidance** — challenges agents to navigate safely through an
  environment filled with obstacles, highlighting spatial awareness and reactive
  control.
- **Trajectory Tracking** — focuses on following predefined paths or trajectories,
  testing precision and stability in continuous control tasks.

These environments serve as practical benchmarks for evaluating different agent
architectures, from tabular methods to deep reinforcement learning approaches.

.. toctree::
    :maxdepth: 1

    environments/dyna_maze
    environments/shortcut_maze
    environments/pendulum
    environments/mecanum_car
    environments/obstacle_avoidance
    environments/trajectory_tracking