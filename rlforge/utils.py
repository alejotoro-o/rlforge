import numpy as np

def argmax(values):
    """
    Return the index of the maximum value in an array, with random tie-breaking.

    Unlike the standard `numpy.argmax`, this function resolves ties by randomly
    selecting among all indices that share the maximum value. This ensures that
    when multiple actions have equal value estimates, exploration is preserved.

    Parameters
    ----------
    values : numpy.ndarray, shape (n_actions,)
        1-D array of values (e.g., action-value estimates).

    Returns
    -------
    index : int
        Index of one of the maximum values, chosen uniformly at random if there
        are multiple maxima.
    """
    top = float("-inf")
    ties = []

    for i in range(len(values)):
        if values[i] > top:
            top = values[i]
            ties = []

        if values[i] == top:
            ties.append(i)
    
    return np.random.choice(ties)


class ExperienceBuffer:
    """
    A fixed-size replay buffer for storing agent experiences.

    The buffer stores transitions of the form `(state, action, reward, terminal, new_state)`.
    When the buffer reaches its maximum capacity, the oldest experience is discarded
    to make room for new ones. This structure is commonly used in reinforcement learning
    to break correlations between consecutive samples and stabilize training.

    Parameters
    ----------
    size : int
        Maximum number of experiences to store in the buffer.
    mini_batch_size : int
        Number of experiences to sample when calling `sample`.

    Attributes
    ----------
    buffer : list
        Internal list storing the experiences.
    size : int
        Maximum capacity of the buffer.
    mini_batch_size : int
        Number of samples returned by `sample`.
    """

    def __init__(self, size, mini_batch_size):
        self.buffer = []
        self.size = size
        self.mini_batch_size = mini_batch_size

    def append(self, state, action, reward, terminal, new_state):
        """
        Add a new experience to the buffer.

        If the buffer is full, the oldest experience is removed before appending
        the new one.

        Parameters
        ----------
        state : object
            Representation of the environment state.
        action : int or float
            Action taken by the agent.
        reward : float
            Reward received after taking the action.
        terminal : bool
            Flag indicating whether the episode ended after this transition.
        new_state : object
            The next state observed after taking the action.
        """
        if len(self.buffer) == self.size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, new_state])

    def sample(self):
        """
        Randomly sample a mini-batch of experiences from the buffer.

        Experiences are sampled uniformly without replacement. This method is
        typically used during training to provide batches of transitions for
        updating the agent.

        Returns
        -------
        list of list
            A list containing `mini_batch_size` experiences, where each experience
            is a list `[state, action, reward, terminal, new_state]`.
        """
        idx = np.random.choice(np.arange(len(self.buffer)), size=self.mini_batch_size)
        return [self.buffer[i] for i in idx]