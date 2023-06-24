import numpy as np

from rlforge.agents import BaseAgent
from rlforge.policies import gaussian
from rlforge.feature_extraction import TileCoder
from rlforge.function_approximation import LinearRegression

class GaussianActorCriticAgent(BaseAgent):

    def __init__(self, actor_step_size, critic_step_size, avg_reward_step_size, dims_ranges,
                 iht_size=4096, num_tilings=8, num_tiles=8, wrap_dims=()):
        self.actor_step_size = actor_step_size
        self.critic_step_size = critic_step_size
        self.avg_reward_step_size = avg_reward_step_size
        self.avg_reward = 0
        self.softmax_probs = None
        self.tile_coder = TileCoder(dims_ranges, iht_size, num_tilings, num_tiles, wrap_dims)
        self.actor = LinearRegression(self.tile_coder.iht_size, 2)
        self.critic = LinearRegression(self.tile_coder.iht_size, 1)

    def start(self, new_state):

        active_tiles = self.tile_coder.get_tiles(new_state)
        values = self.actor.predict(active_tiles, tile_coding_indices=True)
        mu = values[0]
        sigma = np.exp(values[1])
        action = self.select_action(mu, sigma)

        self.prev_tiles = active_tiles
        self.prev_action = action
        self.prev_mu = mu
        self.prev_sigma = sigma

        return action
    
    def step(self, reward, new_state):

        active_tiles = self.tile_coder.get_tiles(new_state)
        values = self.actor.predict(active_tiles, tile_coding_indices=True)
        mu = values[0]
        sigma = np.exp(values[1])

        td_error = self.get_td_error(self.prev_tiles, reward, active_tiles, self.avg_reward)

        self.avg_reward += self.avg_reward_step_size*td_error

        self.critic.update_weights(self.critic_step_size, td_error, state=self.prev_tiles, tile_coding_indices=True)

        grad_mu = td_error*(1/(self.prev_sigma**2))*(self.prev_action - self.prev_mu)
        grad_sigma = td_error*(((self.prev_action - self.prev_mu)**2)/(self.prev_sigma**2) - 1)
        self.actor.update_weights(self.actor_step_size, grad_mu, 0, self.prev_tiles, tile_coding_indices=True)
        self.actor.update_weights(self.actor_step_size, grad_sigma, 1, self.prev_tiles, tile_coding_indices=True)

        action = self.select_action(mu, sigma)

        self.prev_tiles = active_tiles
        self.prev_action = action
        self.prev_mu = mu
        self.prev_sigma = sigma

        return action
    
    def end(self, reward):
        
        state_value = self.critic.predict(self.prev_tiles, tile_coding_indices=True)
        td_error = reward - self.avg_reward - state_value
        self.avg_reward += self.avg_reward_step_size*td_error

        self.critic.update_weights(self.critic_step_size, td_error, state=self.prev_tiles, tile_coding_indices=True)

        grad_mu = td_error*(1/(self.prev_sigma**2))*(self.prev_action - self.prev_mu)
        grad_sigma = td_error*(((self.prev_action - self.prev_mu)**2)/(self.prev_sigma**2) - 1)
        self.actor.update_weights(self.actor_step_size, grad_mu, 0, self.prev_tiles, tile_coding_indices=True)
        self.actor.update_weights(self.actor_step_size, grad_sigma, 1, self.prev_tiles, tile_coding_indices=True)

    
    def select_action(self, mu, sigma):

        action = gaussian(mu, sigma)

        return action
    
    def get_td_error(self, prev_tiles, reward, active_tiles, avg_reward):
        
        target = reward - avg_reward + self.critic.predict(active_tiles, tile_coding_indices=True)
        state_value = self.critic.predict(prev_tiles, tile_coding_indices=True)
        td_error = target - state_value

        return td_error
    
    def reset(self):

        self.actor.reset_weights()
        self.critic.reset_weights()
        self.avg_reward = 0
        