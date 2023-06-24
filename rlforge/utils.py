import numpy as np

def argmax(values):
    """
    Custon argmax function with ramdom tie breaking.

    Args:
        values [1-D Numpy array]: Input array.
    Returns:
        index [integer]: Index of the maximum value in values array.
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
    """
    def __init__(self, size, mini_batch_size):
        self.buffer = []
        self.size = size
        self.mini_batch_size = mini_batch_size

    def append(self, state, action, reward, terminal, new_state):

        if len(self.buffer) == self.size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, new_state])

    def sample(self):

        idx = np.random.choice(np.arange(len(self.buffer)), size=self.mini_batch_size)
        return [self.buffer[i] for i in idx]