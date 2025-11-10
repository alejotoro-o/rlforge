from ..base_agent import BaseAgent
import numpy as np
from ...policies import epsilonGreedy

class SarsaAgent(BaseAgent):
    """
    Tabular agent implementing the **SARSA** algorithm.

    SARSA (State-Action-Reward-State-Action) is an on-policy temporal
    difference learning method. Unlike Q-learning, which updates toward
    the maximum action value in the next state, SARSA updates toward the
    value of the action actually taken under the current policy. This
    makes SARSA sensitive to the agent's exploration strategy.

    Notes
    -----
    - The agent uses an epsilon-greedy policy for action selection.
    - This implementation does not include planning steps; it directly
      inherits from :class:`BaseAgent`.
    """

    def __init__(self, step_size, discount, num_states, num_actions, epsilon=0.1):
        """
        Initialize the SARSA agent.

        Parameters
        ----------
        step_size : float
            Learning rate for Q-value updates.
        discount : float
            Discount factor for future rewards.
        num_states : int
            Number of discrete states in the environment.
        num_actions : int
            Number of discrete actions available to the agent.
        epsilon : float, optional
            Exploration rate for epsilon-greedy action selection (default: 0.1).
        """
        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.reset()

    def start(self, new_state):
        """
        Begin a new episode.

        Selects the first action using the epsilon-greedy policy and
        stores the initial state-action pair.

        Parameters
        ----------
        new_state : int
            The initial state observed from the environment.

        Returns
        -------
        action : int
            The first action selected by the agent.
        """
        action = self.select_action(self.q_values[new_state,:])

        self.prev_action = action
        self.prev_state = new_state

        return action
    
    def step(self, reward, new_state):
        r"""
        Take a step in the environment.

        Updates Q-values using the SARSA update rule, which incorporates
        the action actually taken in the next state.

        Parameters
        ----------
        reward : float
            Reward received from the previous action.
        new_state : int
            The new state observed from the environment.

        Returns
        -------
        action : int
            The next action chosen by the agent.

        Notes
        -----
        - The Q-value update follows:

          .. math::

             Q(s, a) \leftarrow Q(s, a) +
             \alpha \Big[ r + \gamma Q(s', a') - Q(s, a) \Big]

          where :math:`\alpha` is the step size, :math:`\gamma` is the discount
          factor, and :math:`a'` is the action actually taken in the next state.
        """
        action = self.select_action(self.q_values[new_state,:])

        self.q_values[self.prev_state, self.prev_action] += self.step_size * (
            reward + self.discount * self.q_values[new_state, action]
            - self.q_values[self.prev_state, self.prev_action]
        )
        
        self.prev_action = action
        self.prev_state = new_state

        return action

    def end(self, reward):
        """
        Complete an episode.

        Performs a final update to the Q-value of the last state-action pair
        using the terminal reward.

        Parameters
        ----------
        reward : float
            The terminal reward received at the end of the episode.
        """
        self.q_values[self.prev_state, self.prev_action] += self.step_size * (
            reward - self.q_values[self.prev_state, self.prev_action]
        )

    def select_action(self, q_values):
        """
        Select an action using epsilon-greedy exploration.

        Parameters
        ----------
        q_values : numpy.ndarray
            Array of Q-values for the current state.

        Returns
        -------
        action : int
            The chosen action.
        """
        action = epsilonGreedy(q_values, self.epsilon)
        return action       

    def reset(self):
        """
        Reset the agent's internal state.

        Initializes the Q-table to zeros at the start of training or
        between episodes.
        """
        self.q_values = np.zeros((self.num_states, self.num_actions))