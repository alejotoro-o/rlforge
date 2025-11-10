import numpy as np
from rlforge.policies import epsilonGreedy

class BanditAgent:
    """
    Agent for the k-armed bandit problem using epsilon-greedy exploration.

    This agent maintains estimates of action values (Q-values) and
    selects actions according to the epsilon-greedy strategy. After
    receiving a reward, it updates the estimate of the chosen action
    using an incremental sample-average method.

    Parameters
    ----------
    num_actions : int
        Number of available arms (actions).
    epsilon : float, optional (default=0.1)
        Probability of selecting a random action (exploration).
    step_size : float or None, optional
        Constant step size for updates. If None, use 1/n incremental
        average (sample-average method).
    """

    def __init__(self, num_actions, epsilon=0.1, step_size=None):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.step_size = step_size

        # Estimated action values
        self.q_values = np.zeros(num_actions)
        # Count of times each action has been selected
        self.action_counts = np.zeros(num_actions, dtype=int)

    def select_action(self):
        """
        Choose an action using epsilon-greedy exploration.

        Returns
        -------
        action : int
            Index of the chosen action.
        """
        return epsilonGreedy(self.q_values, self.epsilon)

    def update(self, action, reward):
        """
        Update the estimated value of the chosen action.

        Parameters
        ----------
        action : int
            Index of the action taken.
        reward : float
            Reward received from the environment.
        """
        self.action_counts[action] += 1

        if self.step_size is None:
            # Sample-average method
            alpha = 1.0 / self.action_counts[action]
        else:
            # Constant step size
            alpha = self.step_size

        self.q_values[action] += alpha * (reward - self.q_values[action])