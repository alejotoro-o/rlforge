"""
Obstacle Avoidance
==================
"""

import gymnasium as gym
import numpy as np

class ObstacleAvoidance(gym.Env):
    
    """
    
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, x_range = (-2,2), y_range = (-2,2), initial_state=(0,0,0), target=(1,1,0.1), obstacles=[(0.5,0.5,0.2)], dt=0.01):

        """
        
        """
        
        self.initial_state = initial_state
        self.target = target
        self.obstacles = obstacles
        self.x_range = x_range
        self.y_range = y_range
        self.dt = dt

        self.num_actions = 3
        self.actions = (
            (0.5,0,0),
            (0,0,1),
            (0,0,-1)
        )

    def step(self, action):

        """
        
        """

        prev_x, prev_y, prev_theta = self.prev_state[0:3]

        v = np.array(self.actions[action])
        
        dx = v[0]*self.dt
        dy = v[1]*self.dt
        dtheta = v[2]*self.dt

        x = prev_x + (dx*np.cos(prev_theta) - dy*np.sin(prev_theta))
        y = prev_y + (dx*np.sin(prev_theta) + dy*np.sin(prev_theta))
        theta = prev_theta + dtheta

        theta = ((theta + np.pi)%(2*np.pi)) - np.pi

        error = np.sqrt((self.target[0] - x)**2 + (self.target[1] - y)**2)
        heading_error = theta - np.arctan2((self.target[1] - y),(self.target[0] - x))

        reward = -error-np.abs(heading_error)
        is_terminal = False

        # Out of map
        if x < self.x_range[0] or x > self.x_range[1] or y < self.y_range[0] or y > self.y_range[1]:
            reward = -1000
            is_terminal = True

        # Hit obstacle
        for obstacle in self.obstacles:
            if ((x - obstacle[0])**2 + (y - obstacle[1])**2 < obstacle[2]**2):
                reward = -100
                x = prev_x
                y = prev_y
                theta = prev_theta

        # Arrived to target
        if ((x - self.target[0])**2 + (y - self.target[1])**2 < self.target[2]**2):
            reward = 0
            is_terminal = True

        new_state = np.array([x, y, theta, error,heading_error])
        # new_state = np.array([x, y, theta, self.target[0], self.target[1]])

        self.prev_state = new_state

        observation = (new_state, reward, is_terminal, False, None)

        return observation
    
    def reset(self):

        """
        
        """

        error = np.sqrt((self.target[0] - self.initial_state[0])**2 + (self.target[1] - self.initial_state[1])**2)
        heading_error = self.initial_state[2] - np.arctan2((self.target[1] - self.initial_state[1]),(self.target[0] - self.initial_state[0]))

        new_state = np.array([self.initial_state[0], self.initial_state[1], self.initial_state[2], error,heading_error])
        # new_state = np.array([self.initial_state[0], self.initial_state[1], self.initial_state[2], self.target[0], self.target[1]])

        self.prev_state = new_state
        reward = -error-np.abs(heading_error)
        is_terminal = False

        observation = (new_state, reward, is_terminal, False, None) 

        return observation
    
    # def render(self):
    #     pass