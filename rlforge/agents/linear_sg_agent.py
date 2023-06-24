import numpy as np

from rlforge.policies import epsilonGreedy
from rlforge.agents import BaseAgent
from rlforge.feature_extraction import TileCoder
from rlforge.function_approximation import LinearRegression

class LinearQAgent(BaseAgent):

    def __init__(self, step_size, discount, num_actions, dims_ranges, epsilon=0.1, iht_size=4096, num_tilings=8, num_tiles=8, wrap_dims=()):
        self.step_size = step_size
        self.discount = discount
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.tile_coder = TileCoder(dims_ranges, iht_size, num_tilings, num_tiles, wrap_dims)
        self.linear_model = LinearRegression(self.tile_coder.iht_size, self.num_actions)

    def start(self, new_state):

        active_tiles = self.tile_coder.get_tiles(new_state)

        q_values = self.linear_model.predict(active_tiles, tile_coding_indices=True)

        action = self.select_action(q_values)

        self.prev_action = action
        self.prev_tiles = active_tiles

        return action
    
    def step(self, reward, new_state):

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

        q_values = self.linear_model.predict(self.prev_tiles, tile_coding_indices=True)
        td_error_times_gradients = reward - q_values[self.prev_action]
        self.linear_model.update_weights(self.step_size,
                                         td_error_times_gradients,
                                         action=self.prev_action,
                                         state=self.prev_tiles,
                                         tile_coding_indices=True)

    def select_action(self, q_values):

        action = epsilonGreedy(q_values, self.epsilon)

        return action
    
    def get_td_error(self, prev_tiles, prev_action, reward, active_tiles):

        target_q_values = self.discount*np.max(self.linear_model.predict(active_tiles, tile_coding_indices=True))
        q_values = self.linear_model.predict(prev_tiles, tile_coding_indices=True)
        td_error = reward + target_q_values - q_values[prev_action]

        return td_error
    
    def reset(self):

        self.linear_model.reset_weights()
        