import numpy as np
from rlforge.utils import argmax

def epsilonGreedy(q_values, epsilon=0.1):
    """
    Select an action using the epsilon-greedy exploration strategy.

    With probability `epsilon`, a random action is chosen (exploration).
    With probability `1 - epsilon`, the action with the highest estimated
    value in `q_values` is selected (exploitation).

    Parameters
    ----------
    q_values : numpy.ndarray, shape (n_actions,)
        1-D array containing the estimated action values for the current state.
    epsilon : float, optional (default=0.1)
        Probability of selecting a random action instead of the greedy one.

    Returns
    -------
    action : int
        Index of the chosen action.
    """
    num_actions = q_values.shape[0]
    if np.random.uniform() < epsilon:
        action = np.random.randint(num_actions)
    else:
        action = argmax(q_values)

    return action


def softmax(h, temperature=1):
    """
    Compute a softmax policy distribution over actions.

    The softmax policy assigns probabilities to actions based on their
    preferences or values. The `temperature` parameter controls the
    exploration-exploitation trade-off:
    
        - Low temperature (< 1): more greedy, favors high-value actions.
        - High temperature (> 1): more exploratory, probabilities spread more evenly.

    Parameters
    ----------
    h : numpy.ndarray, shape (n_states, n_actions)
        2-D array of action preferences or values. Each row corresponds to a state,
        and each column corresponds to an action.
    temperature : float, optional (default=1)
        Scaling factor that adjusts the entropy of the distribution.

    Returns
    -------
    softmax_probs : numpy.ndarray, shape (n_states, n_actions)
        Probability distribution over actions for each state. Rows sum to 1.
    """
    preferences = h / temperature

    c = np.max(preferences, axis=1).reshape((-1, 1))
    num = np.exp(preferences - c)
    den = np.sum(num, axis=1).reshape((-1, 1))

    softmax_probs = num / den
    softmax_probs = softmax_probs.squeeze()

    return softmax_probs


def gaussian(mu, sigma):
    """
    Sample an action from a Gaussian (normal) distribution.

    This policy is typically used in environments with continuous action spaces,
    where actions are drawn from a distribution parameterized by mean (`mu`)
    and standard deviation (`sigma`).

    Parameters
    ----------
    mu : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    action : float
        A real-valued action sampled from N(mu, sigma^2).
    """
    action = np.random.normal(mu, sigma)

    return action