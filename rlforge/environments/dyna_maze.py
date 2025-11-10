from typing import Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.toy_text.utils import categorical_sample

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class DynaMaze(gym.Env):
    """
    Grid-world environment for testing planning-based reinforcement learning agents.

    The **Dyna Maze** is a 6x9 grid with obstacles and a fixed start state.
    The agent must navigate from the start position to the terminal goal
    located at the top-right corner of the grid. Obstacles block certain
    paths, forcing the agent to explore and plan effectively.

    Features
    --------
    - Discrete state space: each cell in the grid corresponds to a unique state.
    - Discrete action space: four possible moves (UP, RIGHT, DOWN, LEFT).
    - Obstacles: specific grid cells are blocked and cannot be entered.
    - Terminal state: reaching cell (0, 8) ends the episode with reward 1.
    - All other transitions yield reward 0.
    - Compatible with Gymnasium API.

    Notes
    -----
    - Transition probabilities are deterministic (always 1.0).
    - The environment is designed to illustrate the benefits of planning
      algorithms such as Dyna-Q.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        """
        Initialize the Dyna Maze environment.

        Parameters
        ----------
        render_mode : str, optional
            Rendering mode. Supported values are ``"human"`` and ``"rgb_array"``.
        """
        self.shape = (6, 9)
        self.start_state_index = np.ravel_multi_index((2, 0), self.shape)

        self.nS = np.prod(self.shape)
        self.nA = 4

        # Obstacles Location
        self._obstacles = np.zeros(self.shape, dtype=bool)
        self._obstacles[1:4, 1] = True
        self._obstacles[4, 5] = True
        self._obstacles[0:3, 7] = True

        # Transition probabilities
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Initial state distribution (always start at (2, 0))
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
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

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