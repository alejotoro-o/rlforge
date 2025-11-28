import numpy as np

from ...agents import BaseAgent
from ...policies import softmax
from ...feature_extraction import TileCoder
from ...function_approximation import LinearRegression

class SoftmaxActorCriticAgent(BaseAgent):
    """
    Softmax Actor-Critic Agent with average reward baseline.

    This agent implements a vanilla actor-critic algorithm using tile coding
    for feature extraction and linear function approximation for both the
    actor (policy) and critic (value function). The actor updates are guided
    by the temporal-difference (TD) error, and actions are selected using a
    softmax distribution over action preferences.

    Parameters
    ----------
    actor_step_size : float
        Learning rate for the actor updates.
    critic_step_size : float
        Learning rate for the critic updates.
    avg_reward_step_size : float
        Step size for updating the average reward baseline.
    num_actions : int
        Number of discrete actions available in the environment.
    dims_ranges : list of tuple
        Ranges for each state dimension, used by the tile coder.
    temperature : float, optional
        Temperature parameter for softmax exploration (default=1).
    iht_size : int, optional
        Size of the index hash table for tile coding (default=4096).
    num_tilings : int, optional
        Number of tilings used in tile coding (default=8).
    num_tiles : int, optional
        Number of tiles per dimension (default=8).
    wrap_dims : tuple, optional
        Dimensions to wrap in tile coding (default=()).
    """

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
        """
        Begin a new episode.

        Extracts active tiles for the initial state, computes action
        preferences, selects an action using softmax, and caches the
        state-action pair for future updates.

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
        q_values = self.actor.predict(active_tiles, tile_coding_indices=True)
        action = self.select_action(q_values, self.temperature)

        self.prev_tiles = active_tiles
        self.prev_action = action

        return action

    def step(self, reward, new_state):
        """
        Take a step in the environment.

        Updates the actor and critic weights using the TD error and
        average reward baseline, then selects the next action.

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
        """
        Complete an episode.

        Performs a final update of the actor and critic using the terminal
        reward and the last cached state-action pair.

        Parameters
        ----------
        reward : float
            Final reward received at the end of the episode.
        """

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
        """
        Select an action using a softmax distribution.

        Parameters
        ----------
        q_values : array-like
            Action preferences or Q-values for the current state.
        temperature : float
            Temperature parameter controlling exploration.

        Returns
        -------
        int
            The action selected by sampling from the softmax distribution.
        """

        softmax_probs = softmax(q_values.reshape((1,-1)), temperature)
        action = np.random.choice(self.actions, p=softmax_probs)

        self.softmax_probs = softmax_probs

        return action
    
    def get_td_error(self, prev_tiles, reward, active_tiles, avg_reward):
        """
        Compute the temporal-difference (TD) error.

        Calculates the TD error using the reward, average reward baseline,
        the critic's prediction for the next state, and the critic's
        prediction for the previous state.

        Parameters
        ----------
        prev_tiles : array-like
            Active tiles for the previous state.
        reward : float
            Reward received for the transition.
        active_tiles : array-like
            Active tiles for the current state.
        avg_reward : float
            Current estimate of the average reward baseline.

        Returns
        -------
        float
            The computed TD error.
        """

        target = reward - avg_reward + self.critic.predict(active_tiles, tile_coding_indices=True)
        state_value = self.critic.predict(prev_tiles, tile_coding_indices=True)
        td_error = target - state_value

        return td_error

    def reset(self):
        """
        Reset the agent.

        Resets the actor and critic weights to their initial values and
        clears the average reward baseline.
        """

        self.actor.reset_weights()
        self.critic.reset_weights()
        self.avg_reward = 0

