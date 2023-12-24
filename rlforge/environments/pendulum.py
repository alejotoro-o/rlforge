"""


"""

import numpy as np
import gymnasium as gym

class Pendulum(gym.Env):

    """
    
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, continuous = False, g = 9.81, m = float(1/3), l = float(3/2), dt = 0.05):

        """
        
        """

        self.g = g
        self.m = m
        self.l = l
        self.dt = dt
        self.thetap_range = (-2*np.pi,2*np.pi)
        self.continuous = continuous

        if not self.continuous:
            self.valid_actions = (0,1,2)
            self.actions = (-1,0,1)
            self.num_actions = 3
        else:
            self.action_range = (-2,2)

    def step(self, action):

        """
        
        """
        
        prev_theta, prev_thetap = self.prev_state

        if not self.continuous:       
            thetap = prev_thetap + 0.75*(self.actions[action] + self.m*self.l*self.g*np.sin(prev_theta))/(self.m*self.l**2)*self.dt
        else:
            if action < self.action_range[0]:
                action = self.action_range[0]
            if action > self.action_range[1]:
                action = self.action_range[1]
            thetap = prev_thetap + 0.75*(action + self.m*self.l*self.g*np.sin(prev_theta))/(self.m*self.l**2)*self.dt

        theta = prev_theta + thetap*self.dt

        theta = ((theta + np.pi)%(2*np.pi)) - np.pi

        if thetap < self.thetap_range[0] or thetap > self.thetap_range[1]:
            theta = -np.pi
            thetap = 0

        new_state = np.array([theta,thetap])
        reward = -np.abs(theta)
        is_terminal = False

        self.prev_state = new_state

        observation = (new_state, reward, is_terminal, False, None)

        return observation


    def reset(self):

        """
        
        """

        theta = -np.pi
        thetap = 0

        reward = 0
        new_state = np.array([theta,thetap])
        is_terminal = False

        self.prev_state = new_state

        observation = (new_state, reward, is_terminal, False, None)

        return observation

    # def render(self):
    #     """
        
    #     """
    #     pass