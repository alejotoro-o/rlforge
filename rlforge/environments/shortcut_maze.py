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

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, shortcut_episodes = 20, render_mode=None):
        self.shape = (6,9)
        self.start_state_index = np.ravel_multi_index((5, 3), self.shape)
        self.shortcut_episodes = shortcut_episodes
        self.elapsed_episodes = 0

        self.nS = np.prod(self.shape)
        self.nA = 4

        # Obstacles Location
        self._obstacles = np.zeros(self.shape, dtype=bool)
        self._obstacles[3,1:] = True

        self.P1 = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P1[s] = {a: [] for a in range(self.nA)}
            self.P1[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P1[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P1[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P1[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        self._obstacles[3,-1] = False

        self.P2 = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P2[s] = {a: [] for a in range(self.nA)}
            self.P2[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P2[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P2[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P2[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (2, 0)
        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.start_state_index] = 1.0

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode

    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current_position, delta):
        """Determine the outcome for an action. Transition Prob is always 1.0.
        Args:
            current: Current position on the grid as (row, col)
            delta: Change in position for transition
        Returns:
            Tuple of ``(1.0, new_state, reward, terminated)``
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

        return (int(s), r, t, False, {"prob": p})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def render(self):
        pass