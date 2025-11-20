import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PIDStatic(gym.Env):
    """
    The agent supplies PID gains ONLY ONCE (the first action). 
    Later actions are ignored, enforcing a single-run test of static gains.
    """
    def __init__(self, m=1.0, k=1.0, b=1.0, w=1, dt=0.01, episode_len=100):
        super().__init__()
        # Kp, Ki, Kd action space
        self.action_space = spaces.Box(low=np.array([0.0,0.0,0.0]), 
                                       high=np.array([2.0,2.0,2.0]), 
                                       dtype=np.float32)
        # x, xdot, setpoint, integral_err, prev_err
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        self.dt = dt
        self.m, self.k, self.b = m, k, b
        self.w = w
        self.episode_len = episode_len
        self.current_pid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.action_taken = False

    def _get_obs(self):
        return np.array([self.x, self.xdot, self.setpoint, self.integral_err, self.prev_err], dtype=np.float32)

    def step(self, action):
        # The agent's action (Kp, Ki, Kd) is only used for the first step after reset.
        self.current_pid = action
        # if not self.action_taken:
        #     self.current_pid = action
        #     self.action_taken = True
        
        Kp, Ki, Kd = self.current_pid # Use the fixed gains for the whole episode

        # 1. Calculate Error and Control Signal
        self.setpoint = np.cos(self.w*self.t*self.dt)
        err = self.setpoint - self.x
        d_err = (err - self.prev_err) / self.dt
        self.integral_err += err * self.dt
        u = Kp * err + Ki * self.integral_err + Kd * d_err

        # 2. Dynamics Integration (Euler step)
        xddot = (-self.k * self.x - self.b * self.xdot + u) / self.m
        self.xdot += xddot * self.dt
        self.x += self.xdot * self.dt
        self.prev_err = err
        self.t += 1

        # 3. Reward and Termination
        # Densely reward minimizing squared error
        reward = -(err**2) 
        
        # Penalize large control signal (u) to prevent aggressive tuning
        reward -= 0.001 * u**2 

        # Success bonus for being close to setpoint
        if abs(err) < 0.01:
            reward += 1.0 

        terminated = False
        truncated = self.t >= self.episode_len
        
        # Big penalty if system is unstable (e.g., position explodes)
        if abs(self.x) > 10.0:
            reward -= 500.0
            truncated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 0.0
        self.xdot = 0.0
        self.integral_err = 0.0
        self.prev_err = 0.0
        self.setpoint = 1.0
        self.t = 0
        self.current_pid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.action_taken = False
        
        obs = self._get_obs()
        return obs, {}