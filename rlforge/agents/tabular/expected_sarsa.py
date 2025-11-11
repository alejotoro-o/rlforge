from .planning_agent import PlanningAgent
import numpy as np

class ExpectedSarsaAgent(PlanningAgent):
    """
    Tabular agent implementing the **Expected SARSA** algorithm.

    This agent extends :class:`PlanningAgent` and defines the Q-value update
    rule using the Expected SARSA method. Unlike standard SARSA, which updates
    based on the action actually taken, Expected SARSA computes the expected
    value of the next state's Q-values under the current policy. This leads to
    smoother updates and often improved stability.

    Notes
    -----
    - The agent uses an epsilon-greedy policy for action selection.
    - Planning steps can be enabled via the base class to simulate experience
      from the learned model.
    """

    def update_q_values(self, prev_state, prev_action, reward, new_state):
        r"""
        Update Q-values using the Expected SARSA update rule.

        The update is based on the expected value of the next state's Q-values,
        weighted by the probabilities of selecting each action under the
        epsilon-greedy policy.

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
        - The probability distribution `pi` is constructed such that:      
            * Each non-greedy action has probability `epsilon / num_actions`.
            * Greedy actions (those with maximum Q-value) share the remaining 
              probability mass `(1 - epsilon)`.
            
        - The Q-value update follows:

            .. math::
                
                Q(s, a) \leftarrow Q(s, a) +
                \alpha \Big[ r + \gamma \sum_{a'} \pi(a' \mid s') Q(s', a') - Q(s, a) \Big]


            where :math:`\alpha` is the step size, :math:`\gamma` is the discount
            factor, and :math:`\pi(a' \mid s')` is the epsilon-greedy policy.

        """
        q_max = np.max(self.q_values[new_state,:])

        # Probability of taking a non-greedy action
        pi = np.ones(self.num_actions) * (self.epsilon / self.num_actions)

        # Probability of taking the greedy action(s)
        pi += (self.q_values[new_state,:] == q_max) * (
            (1 - self.epsilon) / np.sum(self.q_values[new_state,:] == q_max)
        )

        self.q_values[prev_state, prev_action] += self.step_size * (
            reward + self.discount * np.sum(pi * self.q_values[new_state,:])
            - self.q_values[prev_state, prev_action]
        )