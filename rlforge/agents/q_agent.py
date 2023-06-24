from rlforge.agents.planning_agent import PlanningAgent
import numpy as np

class QAgent(PlanningAgent):

    def update_q_values(self, prev_state, prev_action, reward, new_state):
        
        self.q_values[prev_state, prev_action] += self.step_size*(reward + self.discount*np.max(self.q_values[new_state,:])
                                                                  - self.q_values[prev_state, prev_action])