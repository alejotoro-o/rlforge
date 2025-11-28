import numpy as np

from ...policies import epsilonGreedy
from ...agents import BaseAgent
from ...feature_extraction import TileCoder
from ...function_approximation import LinearRegression

class LinearQAgent(BaseAgent):
    """
    Linear Q-Learning Agent with Tile Coding function approximation.

    This agent implements Q-learning using a linear function approximator
    over tile-coded features. It supports epsilon-greedy exploration and
    incremental weight updates based on temporal-difference (TD) errors.

    Parameters
    ----------
    step_size : float
        Learning rate for weight updates.
    discount : float
        Discount factor (γ) applied to future rewards.
    num_actions : int
        Number of discrete actions available in the environment.
    dims_ranges : list of tuple
        Ranges for each state dimension, used by the tile coder.
    epsilon : float, optional
        Exploration rate for epsilon-greedy policy (default=0.1).
    iht_size : int, optional
        Size of the index hash table for tile coding (default=4096).
    num_tilings : int, optional
        Number of tilings used in tile coding (default=8).
    num_tiles : int, optional
        Number of tiles per dimension (default=8).
    wrap_dims : tuple, optional
        Dimensions to wrap in tile coding (default=()).
    """

    def __init__(self, step_size, discount, num_actions, dims_ranges, epsilon=0.1, iht_size=4096, num_tilings=8, num_tiles=8, wrap_dims=()):
        """
        Initialize the LinearQAgent with tile coding and linear regression model.

        Sets up the tile coder for feature extraction and the linear regression
        model for Q-value approximation.

        See class docstring for parameter descriptions.
        """

        self.step_size = step_size
        self.discount = discount
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.tile_coder = TileCoder(dims_ranges, iht_size, num_tilings, num_tiles, wrap_dims)
        self.linear_model = LinearRegression(self.tile_coder.iht_size, self.num_actions)

    def start(self, new_state):
        """
        Begin a new episode.

        Extracts active tiles for the initial state, computes Q-values,
        selects an action using epsilon-greedy, and caches the state-action
        pair for future updates.

        Parameters
        ----------
        new_state : array-like
            The initial state observed from the environment.

        Returns
        -------
        int
            The action selected by the agent.
        """

        active_tiles = self.tile_coder.get_tiles(new_state)

        q_values = self.linear_model.predict(active_tiles, tile_coding_indices=True)

        action = self.select_action(q_values)

        self.prev_action = action
        self.prev_tiles = active_tiles

        return action
    
    def step(self, reward, new_state):
        """
        Take a step in the environment.

        Updates the linear model weights using the TD error from the previous
        transition, then selects the next action based on the new state.

        Parameters
        ----------
        reward : float
            Reward received from the previous action.
        new_state : array-like
            The new state observed from the environment.

        Returns
        -------
        int
            The next action chosen by the agent.
        """

        active_tiles = self.tile_coder.get_tiles(new_state)
        q_values = self.linear_model.predict(active_tiles, tile_coding_indices=True)
        action = self.select_action(q_values)

        td_error_times_gradients = self.get_td_error(self.prev_tiles, self.prev_action, reward, active_tiles)*1
        self.linear_model.update_weights(self.step_size,
                                         td_error_times_gradients,
                                         action=self.prev_action,
                                         state=self.prev_tiles,
                                         tile_coding_indices=True)

        self.prev_action = action
        self.prev_tiles = active_tiles

        return action
    
    def end(self, reward):
        """
        Complete an episode.

        Performs a final update of the linear model weights using the terminal
        reward and the last cached state-action pair.

        Parameters
        ----------
        reward : float
            Final reward received at the end of the episode.
        """

        q_values = self.linear_model.predict(self.prev_tiles, tile_coding_indices=True)
        td_error_times_gradients = reward - q_values[self.prev_action]
        self.linear_model.update_weights(self.step_size,
                                         td_error_times_gradients,
                                         action=self.prev_action,
                                         state=self.prev_tiles,
                                         tile_coding_indices=True)

    def select_action(self, q_values):
        """
        Select an action using epsilon-greedy policy.

        Parameters
        ----------
        q_values : array-like
            Estimated Q-values for all actions in the current state.

        Returns
        -------
        int
            The action selected by epsilon-greedy exploration.
        """

        action = epsilonGreedy(q_values, self.epsilon)

        return action
    
    def get_td_error(self, prev_tiles, prev_action, reward, active_tiles):
        """
        Compute the temporal-difference (TD) error.

        Calculates the TD error using the reward, the discounted maximum
        Q-value of the next state, and the Q-value of the previous state-action.

        Parameters
        ----------
        prev_tiles : array-like
            Active tiles for the previous state.
        prev_action : int
            Action taken in the previous state.
        reward : float
            Reward received for the transition.
        active_tiles : array-like
            Active tiles for the current state.

        Returns
        -------
        float
            The computed TD error.
        """

        target_q_values = self.discount*np.max(self.linear_model.predict(active_tiles, tile_coding_indices=True))
        q_values = self.linear_model.predict(prev_tiles, tile_coding_indices=True)
        td_error = reward + target_q_values - q_values[prev_action]

        return td_error
    
    def reset(self):
        """
        Reset the agent.

        Resets the weights of the linear regression model to their initial values.
        """

        self.linear_model.reset_weights()
        