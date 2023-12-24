"""

"""

from rlforge.agents.planning_agent import PlanningAgent
import numpy as np

class ExpectedSarsaAgent(PlanningAgent):

    """
    
    """

    def update_q_values(self, prev_state, prev_action, reward, new_state):
        
        q_max = np.max(self.q_values[new_state,:])
        # Probability of taking a non-greedy action
        pi = np.ones(self.num_actions)*(self.epsilon/self.num_actions)
        # Probability of taking the greedy action
        pi += (self.q_values[new_state,:] == q_max)*((1 - self.epsilon)/np.sum(self.q_values[new_state,:] == q_max))

        self.q_values[prev_state, prev_action] += self.step_size*(reward + self.discount*np.sum(pi*self.q_values[new_state,:])
                                                                  - self.q_values[prev_state, prev_action])