import gymnasium as gym
import numpy as np

class MecanumCar(gym.Env):
    """
    Simplified environment simulating a mecanum-wheeled car.

    The **MecanumCar** environment models a robot with mecanum wheels that can
    move forward, rotate, and adjust heading. The agent's objective is to reach
    a target position within a bounded workspace while minimizing positional
    and heading error.

    Features
    --------
    - State space: five-dimensional vector ``[x, y, theta, error, heading_error]``.
      * ``x, y``: car position in the plane.
      * ``theta``: car orientation (wrapped to [-π, π]).
      * ``error``: Euclidean distance to the target.
      * ``heading_error``: difference between car orientation and target heading.
    - Action space: discrete set of velocity commands.
      * Default actions: forward motion, rotate clockwise, rotate counterclockwise.
    - Reward: negative distance and heading error, encouraging the agent to
      approach the target with correct orientation.
    - Terminal condition: reaching the target within a tolerance radius.
    - Penalty: large negative reward if the car leaves the workspace bounds.

    Notes
    -----
    - The environment uses simple kinematic equations with Euler integration.
    - This environment is useful for testing control policies for mobile robots.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, x_range=(-2, 2), y_range=(-2, 2),
                 initial_state=(0, 0, 0), target=(1, 1, 0.1), dt=0.01):
        """
        Initialize the MecanumCar environment.

        Parameters
        ----------
        x_range : tuple(float, float), optional
            Allowed range for the x-coordinate (default: (-2, 2)).
        y_range : tuple(float, float), optional
            Allowed range for the y-coordinate (default: (-2, 2)).
        initial_state : tuple(float, float, float), optional
            Initial state of the car (x, y, theta). Default is (0, 0, 0).
        target : tuple(float, float, float), optional
            Target position and tolerance radius (x, y, radius).
            Default is (1, 1, 0.1).
        dt : float, optional
            Time step for integration (default: 0.01).
        """
        self.initial_state = initial_state
        self.target = target
        self.x_range = x_range
        self.y_range = y_range
        self.dt = dt

        self.num_actions = 3
        self.actions = (
            (0.5, 0, 0),   # forward
            (0, 0, 1),     # rotate counterclockwise
            (0, 0, -1)     # rotate clockwise
        )

    def step(self, action):
        """
        Advance the car dynamics by one time step.

        Parameters
        ----------
        action : int
            Index of the chosen action:
            - 0: forward motion
            - 1: rotate counterclockwise
            - 2: rotate clockwise

        Returns
        -------
        observation : tuple
            A 5-element tuple ``(state, reward, terminated, truncated, info)``:
            - state (numpy.ndarray): [x, y, theta, error, heading_error].
            - reward (float): negative distance and heading error.
            - terminated (bool): True if the target is reached.
            - truncated (bool): always False (no time limit).
            - info (dict or None): unused, set to None.

        Notes
        -----
        - Position is updated using simple kinematics with orientation.
        - Orientation is wrapped to [-π, π].
        - If the car leaves the workspace bounds, it receives a large penalty
          and is reset to the previous state.
        - Reaching the target within the tolerance radius ends the episode.
        """
        prev_x, prev_y, prev_theta = self.prev_state[0:3]

        v = np.array(self.actions[action])

        dx = v[0] * self.dt
        dy = v[1] * self.dt
        dtheta = v[2] * self.dt

        x = prev_x + (dx * np.cos(prev_theta) - dy * np.sin(prev_theta))
        y = prev_y + (dx * np.sin(prev_theta) + dy * np.cos(prev_theta))
        theta = prev_theta + dtheta

        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        error = np.sqrt((self.target[0] - x) ** 2 + (self.target[1] - y) ** 2)
        heading_error = theta - np.arctan2((self.target[1] - y), (self.target[0] - x))

        reward = -error - np.abs(heading_error)
        is_terminal = False

        if x < self.x_range[0] or x > self.x_range[1] or y < self.y_range[0] or y > self.y_range[1]:
            reward = -100
            x, y, theta = prev_x, prev_y, prev_theta

        if ((x - self.target[0]) ** 2 + (y - self.target[1]) ** 2 < self.target[2] ** 2):
            reward = 0
            is_terminal = True

        new_state = np.array([x, y, theta, error, heading_error])
        self.prev_state = new_state

        observation = (new_state, reward, is_terminal, False, None)
        return observation

    def reset(self):
        """
        Reset the car to its initial state.

        Returns
        -------
        observation : tuple
            A 5-element tuple ``(state, reward, terminated, truncated, info)``:
            - state (numpy.ndarray): [x, y, theta, error, heading_error].
            - reward (float): negative distance and heading error at reset.
            - terminated (bool): always False.
            - truncated (bool): always False.
            - info (dict or None): unused, set to None.
        """
        error = np.sqrt((self.target[0] - self.initial_state[0]) ** 2 +
                        (self.target[1] - self.initial_state[1]) ** 2)
        heading_error = self.initial_state[2] - np.arctan2(
            (self.target[1] - self.initial_state[1]),
            (self.target[0] - self.initial_state[0])
        )

        new_state = np.array([
            self.initial_state[0],
            self.initial_state[1],
            self.initial_state[2],
            error,
            heading_error
        ])

        self.prev_state = new_state
        reward = -error - np.abs(heading_error)
        is_terminal = False

        observation = (new_state, reward, is_terminal, False, None)
        return observation

    # def render(self):
    #     """Render the car (not implemented)."""
    #     pass