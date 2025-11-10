from ..base_agent import BaseAgent
import numpy as np
from ...policies import epsilonGreedy

class PlanningAgent(BaseAgent):
    """
    Abstract tabular agent that integrates planning steps to accelerate convergence.

    This agent maintains a Q-table and, optionally, a model of the environment
    for planning-based updates. The planning mechanism allows the agent to
    simulate experience from its learned model, improving sample efficiency
    compared to purely online learning.

    Notes
    -----
    - This is a base class meant to be subclassed. The method
      :meth:`update_q_values` is intentionally left empty and must be
      implemented by derived classes (e.g., SARSA, Q-learning, Expected SARSA).
    - Planning is optional and controlled via the ``planning`` flag and
      ``planning_steps`` parameter.
    """

    def __init__(self, step_size, discount, num_states, num_actions,
                 epsilon=0.1, planning=False, planning_steps=0,
                 exploration_bonus=0):
        """
        Initialize the PlanningAgent.

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
        planning : bool, optional
            Whether to enable planning updates using a learned model (default: False).
        planning_steps : int, optional
            Number of planning updates to perform per real step (default: 0).
        exploration_bonus : float, optional
            Bonus term added during planning to encourage exploration (default: 0).
        """
        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.planning = planning
        self.planning_steps = planning_steps
        if self.planning:
            self.actions = list(range(self.num_actions))
            self.model = {}
            self.exploration_bonus = exploration_bonus
        self.reset()

    def start(self, new_state):
        """
        Begin a new episode.

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
        """
        Take a step in the environment.

        Updates Q-values based on the transition and, if planning is enabled,
        performs additional simulated updates using the learned model.

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
        """
        action = self.select_action(self.q_values[new_state,:])

        self.update_q_values(self.prev_state, self.prev_action, reward, new_state)

        if self.planning:
            self.tau += 1
            self.tau[self.prev_state, self.prev_action] = 0
            self.update_model(self.prev_state, self.prev_action, reward, new_state)
            self.planning_step()
        
        self.prev_action = action
        self.prev_state = new_state

        return action

    def end(self, reward):
        """
        Complete an episode.

        Performs a final update to the Q-value of the last state-action pair.

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
    
    def update_q_values(self, prev_state, prev_action, reward, new_state):
        """
        Update Q-values based on the observed transition.

        This method is intentionally left empty and must be implemented
        by subclasses to define the specific update rule (e.g., SARSA,
        Q-learning, Expected SARSA).

        Parameters
        ----------
        prev_state : int
            The previous state.
        prev_action : int
            The action taken in the previous state.
        reward : float
            The reward received.
        new_state : int
            The new state observed.
        """
        pass

    def update_model(self, prev_state, prev_action, reward, new_state):
        """
        Update the agent's model of the environment.

        Stores the transition (state, action → next state, reward) in the model.
        For unseen states, initializes all actions with default transitions.

        Parameters
        ----------
        prev_state : int
            The previous state.
        prev_action : int
            The action taken in the previous state.
        reward : float
            The reward received.
        new_state : int
            The new state observed.
        """
        if prev_state not in self.model:
            self.model[prev_state] = {prev_action : (new_state, reward)}
            for action in self.actions:
                if action != prev_action:
                    self.model[prev_state][action] = (prev_state, 0)
        else:
            self.model[prev_state][prev_action] = (new_state, reward)

    def planning_step(self):
        """
        Perform planning updates using the learned model.

        Randomly samples stored transitions from the model and applies
        Q-value updates. An exploration bonus can be added to encourage
        revisiting less frequently updated state-action pairs.
        """
        for _ in range(self.planning_steps):
            prev_state = np.random.choice(list(self.model.keys()))
            prev_action = np.random.choice(list(self.model[prev_state].keys()))
            new_state, reward = self.model[prev_state][prev_action]
            reward += self.exploration_bonus * np.sqrt(self.tau[prev_state][prev_action])
            if new_state != -1:
                self.update_q_values(prev_state, prev_action, reward, new_state)
            else:
                self.q_values[prev_state, prev_action] += self.step_size * (
                    reward - self.q_values[prev_state][prev_action]
                )

    def reset(self):
        """
        Reset the agent's internal state.

        Initializes the Q-table to zeros. If planning is enabled, also
        initializes the ``tau`` matrix, which tracks the time since each
        state-action pair was last updated.
        """
        self.q_values = np.zeros((self.num_states, self.num_actions))
        if self.planning:
            self.tau = np.zeros((self.num_states, self.num_actions))