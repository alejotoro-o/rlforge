"""

"""
from ..base_agent import BaseAgent
import numpy as np
from rlforge.policies import epsilonGreedy

class SarsaAgent(BaseAgent):
    """
    
    """
    def __init__(self, step_size, discount, num_states, num_actions, epsilon=0.1):
        """
        
        """
        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.reset()

    def start(self, new_state):
        """
        
        """
        action = self.select_action(self.q_values[new_state,:])

        self.prev_action = action
        self.prev_state = new_state

        return action
    
    def step(self, reward, new_state):

        action = self.select_action(self.q_values[new_state,:])

        self.q_values[self.prev_state, self.prev_action] += self.step_size*(reward + self.discount*self.q_values[new_state, action]
                                                                            - self.q_values[self.prev_state, self.prev_action])
        
        self.prev_action = action
        self.prev_state = new_state

        return action

    def end(self, reward):

        self.q_values[self.prev_state, self.prev_action] += self.step_size*(reward - self.q_values[self.prev_state, self.prev_action])

    def select_action(self, q_values):
        """
        
        """
        action = epsilonGreedy(q_values, self.epsilon)

        return action       

    def reset(self):
        """
        
        """
        self.q_values = np.zeros((self.num_states, self.num_actions))