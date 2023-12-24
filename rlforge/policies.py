import numpy as np
from rlforge.utils import argmax

def epsilonGreedy(q_values, epsilon=0.1):
    """
    The Epsilon-Greedy policy returns the higher reward action with a probability of (epsilon - 1).

    Args:
        q_values [1-D Numpy array]: action values.
        epsilon [float]: probability to explore. Default is 0.1.
    Returns:
        action [integer]: choosen action basen on the action values.
    """
    num_actions = q_values.shape[0]
    if np.random.uniform() < epsilon:
      action = np.random.randint(num_actions)
    else:
      action = argmax(q_values)

    return action

def softmax(h, temperature=1):
    """
    The softmax policy returns the probabilities for taking each action based on the action preferences.

    The policy computes de probabilities by using the normalized softmax function.

    Args:
        h [2-D Numpy array]: action preferences or action values, axis 0 are the entries/states and axis 1 are the values. Temperature [float]: allows adjusting the greediness (or entropy) of the policy. Smaller temperatures (below 1) decreases the entropy, and the agent selects more the greedy action; higher temperatures (above 1) increases the entropy, and the agents selects the actions more uniformly. 
    Returns:
        softmax_probs [2-D Numpy array]: array containing the probabilities of taking each action for each one of the entries/states. Axis 0 are the entries/states and axis 1 are the values.
    """
    preferences = h/temperature
    
    c = np.max(preferences, axis=1)
    c = c.reshape((-1,1))

    num = np.exp(preferences - c)
    den = np.sum(num, axis=1)
    den = den.reshape((-1,1))

    softmax_probs = num/den
    softmax_probs = softmax_probs.squeeze()

    return softmax_probs

def gaussian(mu, sigma):
    """
    The Gaussian policy returns a real value (action) drawn from a gaussian (normal) distribution 
    with mean mu and standard deviation sigma.

    The Gaussian policy is used for tasks that have a continuous action space.

    Args:
        mu [float]: desired mean.
        sigma [float]: desired standard deviation.
    Returns:
        action [float]: sample drawn for a gaussian distribution with mean mu and standard deviation sigma.
    """
    action = np.random.normal(mu, sigma)

    return action