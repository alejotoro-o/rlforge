import numpy as np

class Bandits:
    """
    Basic k-armed bandit environment.

    The k-armed bandit problem is a fundamental reinforcement learning
    setting where an agent repeatedly chooses among k actions ("arms"),
    each associated with an unknown reward distribution. The agent’s
    objective is to maximize cumulative reward by balancing exploration
    and exploitation.

    Parameters
    ----------
    k : int
        Number of arms (actions).
    mean_rewards : array-like, optional
        True mean reward for each arm. If None, sampled from N(0,1).
    reward_std : float, optional
        Standard deviation of reward noise (default: 1.0).

    Attributes
    ----------
    k : int
        Number of arms.
    mean_rewards : numpy.ndarray
        True mean reward for each arm.
    reward_std : float
        Standard deviation of reward noise.
    """

    def __init__(self, k=10, mean_rewards=None, reward_std=1.0):
        self.k = k
        self.mean_rewards = (
            np.array(mean_rewards) if mean_rewards is not None else np.random.randn(k)
        )
        self.reward_std = reward_std

    def pull(self, action):
        """
        Pull the specified arm and observe a reward.

        Parameters
        ----------
        action : int
            Index of the arm to pull (0 ≤ action < k).

        Returns
        -------
        reward : float
            Sampled reward from the arm’s distribution.
        """
        true_mean = self.mean_rewards[action]
        reward = np.random.normal(true_mean, self.reward_std)
        return reward

    def optimal_action(self):
        """
        Return the index of the optimal arm.

        Returns
        -------
        int
            Index of the arm with the highest true mean reward.
        """
        return int(np.argmax(self.mean_rewards))