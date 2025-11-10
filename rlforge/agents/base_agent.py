from abc import abstractmethod

class BaseAgent:
    """
    Abstract base class for all RLForge agents.

    This class defines the standard interface that every agent in RLForge
    must implement. It provides a consistent structure for interacting with
    environments, handling episodes, and selecting actions. By inheriting
    from :class:`BaseAgent`, new agents can be integrated seamlessly into
    the RLForge framework.

    Notes
    -----
    - All methods are abstract and must be implemented by subclasses.
    - The interface is designed to be environment-agnostic, so agents can
      be applied to both discrete and continuous tasks.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the agent.

        This method should set up any internal data structures, hyperparameters,
        or model components required by the agent. For example, a tabular agent
        might initialize a Q-table, while a deep agent might build a neural network.
        """
        pass

    @abstractmethod
    def start(self, state):
        """
        Begin a new episode.

        Parameters
        ----------
        state : object
            The initial state observed from the environment.

        Returns
        -------
        action : int or float or numpy.ndarray
            The first action selected by the agent given the initial state.
        """
        pass

    @abstractmethod
    def step(self, reward, state):
        """
        Take a step in the environment.

        This method is called after the agent receives a reward and the next
        state from the environment. The agent should update its internal
        estimates and return the next action.

        Parameters
        ----------
        reward : float
            The reward received from the previous action.
        state : object
            The new state observed from the environment.

        Returns
        -------
        action : int or float or numpy.ndarray
            The next action chosen by the agent.
        """
        pass
    
    @abstractmethod
    def end(self, reward):
        """
        Complete an episode.

        This method is called when the environment signals that the episode
        has terminated. The agent can use the final reward to update its
        estimates.

        Parameters
        ----------
        reward : float
            The final reward received at the end of the episode.
        """
        pass

    @abstractmethod
    def select_action(self, state):
        """
        Select an action given the current state.

        This method encapsulates the agent's policy. Depending on the
        implementation, it may be deterministic (e.g., greedy) or stochastic
        (e.g., epsilon-greedy, softmax, Gaussian).

        Parameters
        ----------
        state : object
            The current state observed from the environment.

        Returns
        -------
        action : int or float or numpy.ndarray
            The action chosen by the agent.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the agent's internal state.

        This method is called between episodes to clear any temporary
        variables or statistics. It ensures that each episode starts
        from a clean state, without residual information from previous runs.
        """
        pass