from typing import Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.toy_text.utils import categorical_sample

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class ShortcutMaze(gym.Env):
    """
    Grid-world environment with a dynamic shortcut for testing agent adaptability.

    The **Shortcut Maze** is a 6x9 grid where the agent starts at the bottom
    and must reach the terminal goal at the top-right corner. Initially, a wall
    blocks the direct path, forcing the agent to take a longer route. After a
    fixed number of episodes, a shortcut opens, allowing faster convergence if
    the agent adapts its policy.

    Features
    --------
    - Discrete state space: each cell in the grid corresponds to a unique state.
    - Discrete action space: four possible moves (UP, RIGHT, DOWN, LEFT).
    - Obstacles: a row of blocked cells initially prevents direct access.
    - Dynamic environment: after ``shortcut_episodes`` episodes, one obstacle
      is removed, creating a shortcut.
    - Terminal state: reaching cell (0, 8) ends the episode with reward 1.
    - All other transitions yield reward 0.
    - Compatible with Gymnasium API.

    Notes
    -----
    - Transition probabilities are deterministic (always 1.0).
    - This environment is useful for studying how agents adapt to non-stationary
      dynamics and benefit from planning or exploration.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, shortcut_episodes=20, render_mode=None):
        """
        Initialize the Shortcut Maze environment.

        Parameters
        ----------
        shortcut_episodes : int, optional
            Number of episodes before the shortcut opens (default: 20).
        render_mode : str, optional
            Rendering mode. Supported values are ``"human"`` and ``"rgb_array"``.
        """
        self.shape = (6, 9)
        self.start_state_index = np.ravel_multi_index((5, 3), self.shape)
        self.shortcut_episodes = shortcut_episodes
        self.elapsed_episodes = 0

        self.nS = np.prod(self.shape)
        self.nA = 4

        # Obstacles Location
        self._obstacles = np.zeros(self.shape, dtype=bool)
        self._obstacles[3, 1:] = True

        # Transition model before shortcut
        self.P1 = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P1[s] = {a: [] for a in range(self.nA)}
            self.P1[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P1[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P1[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P1[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Open the shortcut (remove one obstacle)
        self._obstacles[3, -1] = False

        # Transition model after shortcut
        self.P2 = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P2[s] = {a: [] for a in range(self.nA)}
            self.P2[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P2[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P2[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P2[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Initial state distribution (always start at (5, 3))
        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.start_state_index] = 1.0

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode

    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """
        Clamp coordinates to remain inside the grid boundaries.

        Parameters
        ----------
        coord : numpy.ndarray
            Candidate coordinates (row, col).

        Returns
        -------
        numpy.ndarray
            Valid coordinates within the grid.
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current_position, delta):
        """
        Compute the transition outcome for a given action.

        Parameters
        ----------
        current_position : tuple(int, int)
            Current grid position (row, col).
        delta : list(int, int)
            Change in position for the action.

        Returns
        -------
        list of tuple
            A list containing a single tuple ``(1.0, new_state, reward, terminated)``.

        Notes
        -----
        - If the new position is an obstacle, the agent remains in the same state
          with reward 0.
        - If the new position is the terminal state (0, 8), the agent receives
          reward 1 and the episode terminates.
        - Otherwise, the agent moves to the new state with reward 0.
        """
        new_position = np.array(current_position) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._obstacles[tuple(new_position)]:
            current_state = np.ravel_multi_index(current_position, self.shape)
            return [(1.0, current_state, 0, False)]

        terminal_state = (0, 8)
        is_terminated = tuple(new_position) == terminal_state
        if is_terminated:
            return [(1.0, new_state, 1, is_terminated)]

        return [(1.0, new_state, 0, False)]

    def step(self, a):
        """
        Execute one step in the environment.

        The transition model depends on whether the shortcut has opened.
        Before ``shortcut_episodes`` episodes, transitions follow ``P1``.
        Afterward, transitions follow ``P2``.

        Parameters
        ----------
        a : int
            Action index (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT).

        Returns
        -------
        observation : int
            The new state index.
        reward : float
            Reward obtained from the transition.
        terminated : bool
            Whether the episode has ended.
        truncated : bool
            Always False (no time limit).
        info : dict
            Additional information, including transition probability.
        """
        if self.elapsed_episodes < self.shortcut_episodes:
            transitions = self.P1[self.s][a]
        else:
            transitions = self.P2[self.s][a]

        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a
        self.elapsed_episodes += 1

        if t:
            self.elapsed_episodes = 0

        if self.render_mode == "human":
            self.render()

        return int(s), r, t, False, {"prob": p}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional options (unused).

        Returns
        -------
        observation : int
            The starting state index.
        info : dict
            Additional information, including probability of starting state.
        """
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}