import numpy as np
import gymnasium as gym

class Pendulum(gym.Env):
    """
    Simplified pendulum environment for reinforcement learning experiments.

    The **Pendulum** environment simulates the dynamics of a pendulum with
    controllable torque. The agent's objective is to keep the pendulum upright
    (theta close to 0). The environment supports both discrete and continuous
    action spaces:

    - **Discrete mode**: three possible torques (-1, 0, +1).
    - **Continuous mode**: torque values in the range [-2, 2].

    Features
    --------
    - State space: two-dimensional vector ``[theta, theta_dot]`` where
      ``theta`` is the pendulum angle (wrapped to [-π, π]) and
      ``theta_dot`` is the angular velocity.
    - Action space: discrete or continuous torques depending on initialization.
    - Reward: negative absolute angle (``-abs(theta)``), encouraging the agent
      to keep the pendulum upright.
    - Deterministic dynamics with simple Euler integration.

    Notes
    -----
    - The pendulum resets to the downward position (theta = -π, theta_dot = 0).
    - Episodes do not terminate naturally (``terminated`` is always False).
    - This environment is lighter and simpler than the standard Gymnasium
      ``Pendulum-v1``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, continuous=False, g=9.81, m=float(1/3), l=float(3/2), dt=0.05):
        """
        Initialize the Pendulum environment.

        Parameters
        ----------
        continuous : bool, optional
            If True, use continuous torque actions in range [-2, 2].
            If False, use discrete actions {-1, 0, +1}. Default is False.
        g : float, optional
            Gravitational acceleration (default: 9.81).
        m : float, optional
            Mass of the pendulum (default: 1/3).
        l : float, optional
            Length of the pendulum (default: 3/2).
        dt : float, optional
            Time step for integration (default: 0.05).
        """
        self.g = g
        self.m = m
        self.l = l
        self.dt = dt
        self.thetap_range = (-2 * np.pi, 2 * np.pi)
        self.continuous = continuous

        if not self.continuous:
            self.valid_actions = (0, 1, 2)
            self.actions = (-1, 0, 1)
            self.num_actions = 3
        else:
            self.action_range = (-2, 2)

    def step(self, action):
        """
        Advance the pendulum dynamics by one time step.

        Parameters
        ----------
        action : int or float
            - If discrete mode: index of the action (0, 1, 2) corresponding
              to torques -1, 0, +1.
            - If continuous mode: torque value in range [-2, 2].

        Returns
        -------
        observation : tuple
            A 5-element tuple ``(state, reward, terminated, truncated, info)``:
            - state (numpy.ndarray): [theta, theta_dot] after the step.
            - reward (float): negative absolute angle.
            - terminated (bool): always False (no terminal state).
            - truncated (bool): always False (no time limit).
            - info (dict or None): unused, set to None.

        Notes
        -----
        - The angle ``theta`` is wrapped to [-π, π].
        - If angular velocity exceeds the allowed range, the pendulum resets
          to the downward position.
        """
        prev_theta, prev_thetap = self.prev_state

        if not self.continuous:
            thetap = prev_thetap + 0.75 * (
                self.actions[action] + self.m * self.l * self.g * np.sin(prev_theta)
            ) / (self.m * self.l ** 2) * self.dt
        else:
            if action < self.action_range[0]:
                action = self.action_range[0]
            if action > self.action_range[1]:
                action = self.action_range[1]
            thetap = prev_thetap + 0.75 * (
                action + self.m * self.l * self.g * np.sin(prev_theta)
            ) / (self.m * self.l ** 2) * self.dt

        theta = prev_theta + thetap * self.dt
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        if thetap < self.thetap_range[0] or thetap > self.thetap_range[1]:
            theta = -np.pi
            thetap = 0

        new_state = np.array([theta, thetap])
        reward = -np.abs(theta)
        is_terminal = False

        self.prev_state = new_state

        observation = (new_state, reward, is_terminal, False, None)
        return observation

    def reset(self):
        """
        Reset the pendulum to its initial state.

        Returns
        -------
        observation : tuple
            A 5-element tuple ``(state, reward, terminated, truncated, info)``:
            - state (numpy.ndarray): initial state [-π, 0].
            - reward (float): always 0 at reset.
            - terminated (bool): always False.
            - truncated (bool): always False.
            - info (dict or None): unused, set to None.
        """
        theta = -np.pi
        thetap = 0

        reward = 0
        new_state = np.array([theta, thetap])
        is_terminal = False

        self.prev_state = new_state

        observation = (new_state, reward, is_terminal, False, None)
        return observation

    # def render(self):
    #     """
    #     Render the pendulum (not implemented).
    #     """
    #     pass