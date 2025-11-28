import numpy as np

from ...agents import BaseAgent
from ...policies import gaussian
from ...feature_extraction import TileCoder
from ...function_approximation import LinearRegression

class GaussianActorCriticAgent(BaseAgent):
    """
    Gaussian Actor-Critic Agent with average reward baseline.

    This agent implements a continuous-action actor-critic algorithm using
    tile coding for feature extraction and linear function approximation for
    both the actor (policy) and critic (value function). The actor outputs
    the mean and log standard deviation of a Gaussian distribution, from
    which actions are sampled. Updates are guided by the temporal-difference
    (TD) error and an average reward baseline.

    Parameters
    ----------
    actor_step_size : float
        Learning rate for the actor updates.
    critic_step_size : float
        Learning rate for the critic updates.
    avg_reward_step_size : float
        Step size for updating the average reward baseline.
    dims_ranges : list of tuple
        Ranges for each state dimension, used by the tile coder.
    iht_size : int, optional
        Size of the index hash table for tile coding (default=4096).
    num_tilings : int, optional
        Number of tilings used in tile coding (default=8).
    num_tiles : int, optional
        Number of tiles per dimension (default=8).
    wrap_dims : tuple, optional
        Dimensions to wrap in tile coding (default=()).
    """

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
        """
        Begin a new episode.

        Extracts active tiles for the initial state, computes the Gaussian
        parameters (mean and standard deviation), samples an action, and
        caches the state-action pair for future updates.

        Parameters
        ----------
        new_state : array-like
            The initial state observed from the environment.

        Returns
        -------
        float
            The continuous action selected by the agent.
        """

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
        """
        Take a step in the environment.

        Updates the actor and critic weights using the TD error and average
        reward baseline, then selects the next action by sampling from the
        Gaussian distribution defined by the actor.

        Parameters
        ----------
        reward : float
            Reward received from the previous action.
        new_state : array-like
            The new state observed from the environment.

        Returns
        -------
        float
            The next continuous action chosen by the agent.
        """

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

        grad_mu = td_error*(1/(self.prev_sigma**2))*(self.prev_action - self.prev_mu)
        grad_sigma = td_error*(((self.prev_action - self.prev_mu)**2)/(self.prev_sigma**2) - 1)
        self.actor.update_weights(self.actor_step_size, grad_mu, 0, self.prev_tiles, tile_coding_indices=True)
        self.actor.update_weights(self.actor_step_size, grad_sigma, 1, self.prev_tiles, tile_coding_indices=True)

    
    def select_action(self, mu, sigma):
        """
        Select an action by sampling from a Gaussian distribution.

        Parameters
        ----------
        mu : float
            Mean of the Gaussian distribution.
        sigma : float
            Standard deviation of the Gaussian distribution.

        Returns
        -------
        float
            The sampled continuous action.
        """

        action = gaussian(mu, sigma)

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
        