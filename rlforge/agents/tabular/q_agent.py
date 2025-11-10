from .planning_agent import PlanningAgent
import numpy as np

class QAgent(PlanningAgent):
    """
    Tabular agent implementing the **Q-learning** algorithm.

    This agent extends :class:`PlanningAgent` and defines the Q-value update
    rule using the off-policy Q-learning method. Unlike SARSA, which updates
    based on the action actually taken, Q-learning updates toward the maximum
    estimated action value in the next state. This makes Q-learning an
    off-policy algorithm that learns the optimal greedy policy regardless of
    the agent's current behavior.

    Notes
    -----
    - The agent uses an epsilon-greedy policy for action selection.
    - Planning steps can be enabled via the base class to simulate experience
      from the learned model.
    """

    def update_q_values(self, prev_state, prev_action, reward, new_state):
        r"""
        Update Q-values using the Q-learning update rule.

        The update is based on the maximum Q-value in the next state,
        rather than the expected value under the current policy. This
        makes Q-learning an off-policy method.

        Parameters
        ----------
        prev_state : int
            The previous state index.
        prev_action : int
            The action taken in the previous state.
        reward : float
            The reward received after taking the action.
        new_state : int
            The new state index observed after the transition.

        Notes
        -----
        - The Q-value update follows:

          .. math::

             Q(s, a) \leftarrow Q(s, a) +
             \alpha \Big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Big]

          where :math:`\alpha` is the step size, :math:`\gamma` is the discount
          factor, and :math:`\max_{a'} Q(s', a')` is the maximum action value
          in the next state.
        """
        self.q_values[prev_state, prev_action] += self.step_size * (
            reward + self.discount * np.max(self.q_values[new_state,:])
            - self.q_values[prev_state, prev_action]
        )