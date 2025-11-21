from typing import Optional, Union, Tuple, Dict, Any
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# Actions
LEFT = 0
RIGHT = 1

class ShortCorridor(gym.Env):
    """
    The Short Corridor Gridworld (Sutton & Barto Example 13.1).

    The environment has 4 states (0, 1, 2, 3), where 3 is the terminal state.
    - Start state is S=0.
    - Reward is -1 per step until the terminal state is reached. (Note: The description 
      says "reward is 1 per step, as usual" for the example, but the goal is to minimize 
      steps, which is traditionally achieved by R=-1 per step. We use R=-1 for the 
      standard shortest-path/episodic formulation.)
    - Actions are LEFT (0) and RIGHT (1).
    - Transitions:
        - State 0: LEFT -> 0 (stay), RIGHT -> 1
        - State 1: LEFT -> 0, RIGHT -> 2 (Reversed!)
        - State 2: LEFT -> 1, RIGHT -> 3 (Terminal)
    
    The observation returned is controlled by `observation_type`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, observation_type: str = 'tabular', render_mode: Optional[str] = None):
        super().__init__()
        
        # State space: 4 states (0, 1, 2, 3) where 3 is terminal
        self.num_states = 4 
        self.start_state = 0
        self.terminal_state = 3

        # Action space: LEFT (0) and RIGHT (1)
        self.action_space = spaces.Discrete(2)
        
        self.observation_type = observation_type.lower()
        
        if self.observation_type == 'tabular':
            # Tabular observation: state index (0, 1, 2, 3)
            self.observation_space = spaces.Discrete(self.num_states)
        elif self.observation_type == 'feature':
            # Feature observation for function approximation: 
            # Vector [1, 0] or [0, 1] representing the action taken.
            self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown observation_type: {observation_type}. Must be 'tabular' or 'feature'.")

        self.render_mode = render_mode
        self.s = self.start_state # Current state index

        # Store the transitions P[state][action] -> (prob, next_state, reward, terminated)
        self.P = {s: {a: [] for a in self.actions} for s in range(self.num_states)}
        self._build_transitions()


    @property
    def actions(self):
        """Helper property for action indices."""
        return list(range(self.action_space.n))

    def _build_transitions(self):
        """Defines the transition dynamics based on the Short Corridor problem."""
        
        # State 0 (Non-terminal)
        self.P[0][LEFT] = [(1.0, 0, -1.0, False)]  # Stay at 0
        self.P[0][RIGHT] = [(1.0, 1, -1.0, False)] # Move to 1

        # State 1 (Non-terminal - Reversed actions)
        self.P[1][LEFT] = [(1.0, 2, -1.0, False)]  # LEFT moves RIGHT to 2
        self.P[1][RIGHT] = [(1.0, 0, -1.0, False)] # RIGHT moves LEFT to 0

        # State 2 (Non-terminal)
        self.P[2][LEFT] = [(1.0, 1, -1.0, False)]  # Move to 1
        self.P[2][RIGHT] = [(1.0, 3, -1.0, True)]  # Move to 3 (Terminal, Reward 0 for final transition)
        
        # State 3 (Terminal)
        for a in self.actions:
            self.P[3][a] = [(1.0, 3, 0.0, True)] # No movement, no reward once terminal

    def _get_obs(self, action: Optional[int] = None) -> Union[int, np.ndarray]:
        """
        Returns the observation based on the configured type.
        
        - 'tabular': Returns the state index.
        - 'feature': Returns the feature vector x(s, a). Note that the feature vector 
          is solely dependent on the *action taken* for this specific problem (perceptual aliasing).
        """
        if self.observation_type == 'tabular':
            return self.s
        
        # 'feature' observation: x(s, a)
        # x(s, right) = [1, 0], x(s, left) = [0, 1]
        if action == RIGHT:
            return np.array([1.0, 0.0], dtype=np.float32)
        elif action == LEFT:
            return np.array([0.0, 1.0], dtype=np.float32)
        else:
            # Should not happen on start(), but return zero vector as a safe default
            return np.array([0.0, 0.0], dtype=np.float32)


    def step(self, a: int) -> Tuple[Union[int, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Performs a single step in the environment.
        
        Note: The observation returned by step is based on the *action taken* to 
        get to the current state, as described in the problem's feature definition:
        x(s, right) = [1, 0], x(s, left) = [0, 1], for all s.
        """
        
        # Lookup the transition (prob, next_state, reward, terminated)
        # Since prob is always 1.0, we can just grab the first (and only) entry
        prob, next_s, r, terminated = self.P[self.s][a][0]
        
        self.s = next_s
        
        # The feature-based observation depends on the *action* (a), not the state (self.s)
        # We pass the action (a) to _get_obs to produce the feature x(s, a)
        observation = self._get_obs(action=a) 

        return observation, r, terminated, False, {"prob": prob, "true_state": self.s}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Union[int, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to its initial state (s=0).
        """
        super().reset(seed=seed)
        self.s = self.start_state
        
        # At start, no action has been taken. _get_obs returns tabular state 
        # or a zero-vector if feature-based (as there's no action to base it on).
        observation = self._get_obs()

        return observation, {"true_state": self.s}
    
    # --- Rendering (Simplified for a console view) ---
    def render(self):
        if self.render_mode == "human":
            corridor = ['[S]','[ ]','[ ]','[G]']
            corridor[self.s] = '[X]' # Mark current position
            
            print(f"Corridor: {' '.join(corridor)} (State: {self.s})")

    def close(self):
        # No resources to close
        pass