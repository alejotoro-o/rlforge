import numpy as np

from rlforge.agents import BaseAgent
from rlforge.policies import softmax
from rlforge.feature_extraction import TileCoder
from rlforge.function_approximation import LinearRegression

class SoftmaxActorCriticAgent(BaseAgent):

    def __init__(self, actor_step_size, critic_step_size, avg_reward_step_size, num_actions, dims_ranges,
                 temperature=1, iht_size=4096, num_tilings=8, num_tiles=8, wrap_dims=()):
        self.actor_step_size = actor_step_size
        self.critic_step_size = critic_step_size
        self.avg_reward_step_size = avg_reward_step_size
        self.avg_reward = 0
        self.softmax_probs = None
        self.num_actions = num_actions
        self.actions = list(range(self.num_actions))
        self.temperature = temperature
        self.tile_coder = TileCoder(dims_ranges, iht_size, num_tilings, num_tiles, wrap_dims)
        self.actor = LinearRegression(self.tile_coder.iht_size, self.num_actions)
        self.critic = LinearRegression(self.tile_coder.iht_size, 1)

    def start(self, new_state):

        active_tiles = self.tile_coder.get_tiles(new_state)
        q_values = self.actor.predict(active_tiles, tile_coding_indices=True)
        action = self.select_action(q_values, self.temperature)

        self.prev_tiles = active_tiles
        self.prev_action = action

        return action

    def step(self, reward, new_state):
        
        active_tiles = self.tile_coder.get_tiles(new_state)
        q_values = self.actor.predict(active_tiles, tile_coding_indices=True)

        td_error = self.get_td_error(self.prev_tiles, reward, active_tiles, self.avg_reward)

        self.avg_reward += self.avg_reward_step_size*td_error

        self.critic.update_weights(self.critic_step_size, td_error, state=self.prev_tiles, tile_coding_indices=True)

        for a in self.actions:
            if a == self.prev_action:
                grad = td_error*(1 - self.softmax_probs[a])
                self.actor.update_weights(self.actor_step_size, grad, a, self.prev_tiles, tile_coding_indices=True)
            else:
                grad = td_error*(0 - self.softmax_probs[a])
                self.actor.update_weights(self.actor_step_size, grad, a, self.prev_tiles, tile_coding_indices=True)

        action = self.select_action(q_values, self.temperature)

        self.prev_tiles = active_tiles
        self.prev_action = action

        return action

    def end(self, reward):
        
        state_value = self.critic.predict(self.prev_tiles, tile_coding_indices=True)
        td_error = reward - self.avg_reward - state_value
        self.avg_reward += self.avg_reward_step_size*td_error

        self.critic.update_weights(self.critic_step_size, td_error, state=self.prev_tiles, tile_coding_indices=True)

        for a in self.actions:
            if a == self.prev_action:
                grad = td_error*(1 - self.softmax_probs[a])
                self.actor.update_weights(self.actor_step_size, grad, a, self.prev_tiles, tile_coding_indices=True)
            else:
                grad = td_error*(0 - self.softmax_probs[a])
                self.actor.update_weights(self.actor_step_size, grad, a, self.prev_tiles, tile_coding_indices=True)

    def select_action(self, q_values, temperature):
        
        softmax_probs = softmax(q_values.reshape((1,-1)), temperature)
        action = np.random.choice(self.actions, p=softmax_probs)

        self.softmax_probs = softmax_probs

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

