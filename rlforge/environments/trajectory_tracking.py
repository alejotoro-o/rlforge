import gymnasium as gym
import numpy as np

class TrajectoryTracking(gym.Env):
    """
    Mobile robot environment for trajectory tracking tasks.

    The **TrajectoryTracking** environment models a robot navigating in a
    bounded 2D workspace. The agent's objective is to follow a sequence of
    waypoints (trajectory) while avoiding obstacles and staying within the
    workspace limits.

    Features
    --------
    - State space: three-dimensional vector ``[x, y, theta]``.
      * ``x, y``: robot position in the plane.
      * ``theta``: robot orientation (wrapped to [-π, π]).
    - Action space: discrete set of velocity commands.
      * Forward motion.
      * Rotate counterclockwise.
      * Rotate clockwise.
    - Reward:
      * Negative distance and heading error to the current waypoint.
      * Large penalty if leaving the workspace.
      * Penalty if colliding with an obstacle.
      * Positive reward when reaching waypoints (scaled by waypoint index).
    - Terminal condition: reaching the final waypoint in the trajectory.
    - Compatible with Gymnasium API.

    Notes
    -----
    - Obstacles are defined as circles with center coordinates and radius.
    - The environment uses simple kinematic equations with Euler integration.
    - Waypoints are reached when the robot is within a distance threshold ``d_min``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, x_range=(-2, 2), y_range=(-2, 2),
                 initial_state=(0, 0, 0), trajectory=[(1, 1)],
                 d_min=0.05, obstacles=[], dt=0.01):
        """
        Initialize the TrajectoryTracking environment.

        Parameters
        ----------
        x_range : tuple(float, float), optional
            Allowed range for the x-coordinate (default: (-2, 2)).
        y_range : tuple(float, float), optional
            Allowed range for the y-coordinate (default: (-2, 2)).
        initial_state : tuple(float, float, float), optional
            Initial state of the robot (x, y, theta). Default is (0, 0, 0).
        trajectory : list of tuple(float, float), optional
            Sequence of waypoints (x, y) to be followed. Default is [(1, 1)].
        d_min : float, optional
            Minimum distance threshold to consider a waypoint reached (default: 0.05).
        obstacles : list of tuple(float, float, float), optional
            List of circular obstacles defined by (x, y, radius). Default is [].
        dt : float, optional
            Time step for integration (default: 0.01).
        """
        self.initial_state = initial_state
        self.trajectory = trajectory
        self.obstacles = obstacles
        self.d_min = d_min
        self.waypoint = 0
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
        Advance the robot dynamics by one time step.

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
            - state (numpy.ndarray): [x, y, theta].
            - reward (float): reward based on distance, heading error, penalties, and waypoint progress.
            - terminated (bool): True if the final waypoint is reached.
            - truncated (bool): always False (no time limit).
            - info (dict or None): unused, set to None.

        Notes
        -----
        - Position is updated using simple kinematics with orientation.
        - Orientation is wrapped to [-π, π].
        - Leaving the workspace yields a large penalty and resets to the previous state.
        - Colliding with an obstacle yields a penalty and resets to the previous state.
        - Reaching a waypoint yields a positive reward proportional to the waypoint index.
        - Reaching the final waypoint ends the episode.
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

        error = np.sqrt((self.trajectory[self.waypoint][0] - x) ** 2 +
                        (self.trajectory[self.waypoint][1] - y) ** 2)
        heading_error = theta - np.arctan2(
            (self.trajectory[self.waypoint][1] - y),
            (self.trajectory[self.waypoint][0] - x)
        )

        reward = -error - np.abs(heading_error)
        is_terminal = False

        # Out of map
        if x < self.x_range[0] or x > self.x_range[1] or y < self.y_range[0] or y > self.y_range[1]:
            reward = -1000
            x, y, theta = prev_x, prev_y, prev_theta

        # Hit obstacle
        for obstacle in self.obstacles:
            if ((x - obstacle[0]) ** 2 + (y - obstacle[1]) ** 2 < obstacle[2] ** 2):
                reward = -100
                x, y, theta = prev_x, prev_y, prev_theta

        # Arrived to waypoint
        if ((x - self.trajectory[self.waypoint][0]) ** 2 +
            (y - self.trajectory[self.waypoint][1]) ** 2 < self.d_min ** 2):
            self.waypoint = self.waypoint + 1
            reward = 100 * self.waypoint

            # Arrived to final waypoint
            if self.waypoint == len(self.trajectory):
                is_terminal = True

        new_state = np.array([x, y, theta])
        self.prev_state = new_state

        observation = (new_state, reward, is_terminal, False, None)
        return observation

    def reset(self):
        """
        Reset the robot to its initial state and restart the trajectory.

        Returns
        -------
        observation : tuple
            A 5-element tuple ``(state, reward, terminated, truncated, info)``:
            - state (numpy.ndarray): [x, y, theta].
            - reward (float): initial reward (set to -1).
            - terminated (bool): always False.
            - truncated (bool): always False.
            - info (dict or None): unused, set to None.
        """
        self.waypoint = 0

        new_state = np.array([
            self.initial_state[0],
            self.initial_state[1],
            self.initial_state[2]
        ])

        self.prev_state = new_state
        reward = -1
        is_terminal = False

        observation = (new_state, reward, is_terminal, False, None)
        return observation

    # def render(self):
    #     """Render the environment (not implemented)."""
    #     pass