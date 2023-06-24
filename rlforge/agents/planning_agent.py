from rlforge.agents.base_agent import BaseAgent
import numpy as np
from rlforge.policies import epsilonGreedy

class PlanningAgent(BaseAgent):
    """
    
    """
    def __init__(self, step_size, discount, num_states, num_actions, epsilon=0.1, planning=False, planning_steps=0,exploration_bonus=0):
        """
        
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
        
        """
        action = self.select_action(self.q_values[new_state,:])

        self.prev_action = action
        self.prev_state = new_state

        return action
    
    def step(self, reward, new_state):

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

        self.q_values[self.prev_state, self.prev_action] += self.step_size*(reward - self.q_values[self.prev_state, self.prev_action])

    def select_action(self, q_values):
        """
        
        """
        action = epsilonGreedy(q_values, self.epsilon)

        return action       
    
    def update_q_values(self, prev_state, prev_action, reward, new_state):
        pass

    def update_model(self, prev_state, prev_action, reward, new_state):
        if prev_state not in self.model:
            self.model[prev_state] = {prev_action : (new_state, reward)}
            for action in self.actions:
                if action != prev_action:
                    self.model[prev_state][action] = (prev_state, 0)
        else:
            self.model[prev_state][prev_action] = (new_state, reward)

    def planning_step(self):
        for _ in range(self.planning_steps):
            prev_state = np.random.choice(list(self.model.keys()))
            prev_action = np.random.choice(list(self.model[prev_state].keys()))
            new_state, reward = self.model[prev_state][prev_action]
            reward += self.exploration_bonus*np.sqrt(self.tau[prev_state][prev_action])
            if new_state != -1:
                self.update_q_values(prev_state, prev_action, reward, new_state)
            else:
                self.q_values[prev_state,prev_action] += self.step_size*(reward - self.q_values[prev_state][prev_action])

    def reset(self):
        """
        
        """
        self.q_values = np.zeros((self.num_states, self.num_actions))
        if self.planning:
            self.tau = np.zeros((self.num_states, self.num_actions))