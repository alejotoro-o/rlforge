import numpy as np

from ...agents import BaseAgent 
from ...policies import softmax 
from ...feature_extraction import TileCoder
from ...function_approximation import LinearRegression 

class REINFORCEAgent(BaseAgent):
    """
    REINFORCE (Monte Carlo Policy Gradient) Agent for discrete action spaces.

    This agent implements the REINFORCE algorithm using linear function
    approximation over tile-coded features. It supports both vanilla
    REINFORCE and REINFORCE with a baseline.

    - Vanilla REINFORCE (baseline=False): Uses Monte Carlo return G_t as the update factor.
    - REINFORCE with Baseline (baseline=True): Uses advantage A_t = G_t - V(s_t, w) as the update factor.

    Parameters
    ----------
    num_actions : int
        Number of discrete actions available in the environment.
    dims_ranges : list of tuple
        Ranges for each state dimension, used by the tile coder.
    alpha_theta : float, optional
        Policy step size (default=2e-3).
    alpha_w : float, optional
        Baseline step size (default=2e-3). Only used if baseline=True.
    discount : float, optional
        Discount factor γ applied to future rewards (default=0.99).
    iht_size : int, optional
        Size of the index hash table for tile coding (default=4096).
    num_tilings : int, optional
        Number of tilings used in tile coding (default=8).
    num_tiles : int, optional
        Number of tiles per dimension (default=8).
    baseline : bool, optional
        Whether to use a baseline value function (default=False).
    wrap_dims : tuple, optional
        Dimensions to wrap in tile coding (default=()).
    """


    def __init__(self, 
                 num_actions, 
                 dims_ranges,
                 alpha_theta=2e-3,          # Policy step size (alpha^theta)
                 alpha_w=2e-3,              # Baseline step size (alpha^w) - Only used if baseline=True 
                 discount=0.99,
                 iht_size=4096, 
                 num_tilings=8, 
                 num_tiles=8, 
                 baseline=False, 
                 wrap_dims=()):  

        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.discount = discount
        
        # Ensure num_actions is stored as an integer
        self.num_actions = int(num_actions)
        
        # Use np.arange for a guaranteed correct list of actions as an array of ints
        self.actions = np.arange(self.num_actions, dtype=int)
        
        self.use_baseline = baseline

        # Feature Extractor
        self.tile_coder = TileCoder(dims_ranges, iht_size, num_tilings, num_tiles, wrap_dims)
        
        # Function Approximators
        self.actor = LinearRegression(self.tile_coder.iht_size, self.num_actions)
        if self.use_baseline:
            self.baseline_func = LinearRegression(self.tile_coder.iht_size, 1)

        # Trajectory storage for Monte Carlo update
        self.trajectory = [] # Stores (tiles_t, action_t, reward_t+1)
        self.prev_tiles = None 
        self.prev_action = None
    
    def start(self, state):
        """
        Begin a new episode.

        Extracts active tiles for the initial state, computes action
        preferences, applies softmax to obtain action probabilities,
        samples an action, and caches the state-action pair.

        Parameters
        ----------
        state : np.ndarray
            The initial state or feature vector.

        Returns
        -------
        int
            The chosen action.
        """

        self.trajectory = []
        
        # 1. Feature extraction
        active_tiles = self.tile_coder.get_tiles(state)
        
        # 2. Predict action preferences h(s, a). Result is typically (1, num_actions)
        h_values = self.actor.predict(active_tiles, tile_coding_indices=True)
        
        # --- CRITICAL FIX: Softmax requires 2D input (N_states, N_actions) ---
        # 3. Ensure input to softmax is 2D (1, N) for compatibility with its internal axis=1 operation
        h_values_2d = h_values.reshape((1, -1))
        
        # 4. Compute softmax probabilities
        # Softmax requires the 2D shape to correctly apply np.max/np.sum over axis=1.
        softmax_result = softmax(h_values_2d)

        # 5. Flatten the result back to 1D (N,) for np.random.choice 'p' argument
        softmax_probs = softmax_result.flatten()

        # 6. Sample action
        action = np.random.choice(self.actions, p=softmax_probs)
        
        # 7. Cache for step/end
        self.prev_tiles = active_tiles
        self.prev_action = action
        
        return action

    def step(self, reward, state):
        """
        Perform a step transition.

        Stores the transition (S_t, A_t, R_{t+1}), computes action
        preferences for the new state, applies softmax, samples the
        next action, and caches it.

        Parameters
        ----------
        reward : float
            Reward received from the previous action.
        state : np.ndarray
            The new state observed from the environment.

        Returns
        -------
        int
            The next action chosen by the agent.
        """

        # Store the current transition (s_t, a_t, r_{t+1})
        self.trajectory.append((self.prev_tiles, self.prev_action, reward))
        
        # 1. Feature extraction
        active_tiles = self.tile_coder.get_tiles(state)
        
        # 2. Predict action preferences h(s, a)
        h_values = self.actor.predict(active_tiles, tile_coding_indices=True)
        
        # --- CRITICAL FIX: Softmax requires 2D input (N_states, N_actions) ---
        # 3. Ensure input to softmax is 2D (1, N)
        h_values_2d = h_values.reshape((1, -1))
        
        # 4. Compute softmax probabilities
        softmax_result = softmax(h_values_2d)

        # 5. Flatten the result back to 1D (N,)
        softmax_probs = softmax_result.flatten()

        # 6. Sample action
        action = np.random.choice(self.actions, p=softmax_probs)
        
        # 7. Cache for next step
        self.prev_tiles = active_tiles
        self.prev_action = action
        
        return action
    
    def end(self, reward):
        """
        Process the final reward and execute the Monte Carlo update.

        Computes returns G_t for the entire episode, calculates the
        update factor (G_t or A_t depending on baseline usage), updates
        the baseline function if enabled, and applies the policy gradient
        update to the actor.

        Parameters
        ----------
        reward : float
            Final reward received at the end of the episode.
        """

        # Store the final transition
        self.trajectory.append((self.prev_tiles, self.prev_action, reward))

        # 1. Calculate Returns (G) for the whole episode (Monte Carlo)
        G = 0.0
        
        # Loop backwards through the trajectory to calculate returns
        for t in reversed(range(len(self.trajectory))):
            tiles_t, action_t, reward_t_plus_1 = self.trajectory[t]
            
            # Update Return G_t
            G = reward_t_plus_1 + self.discount * G
            
            # --- 2. Calculate Factor (G_t or A_t) ---
            A_t = G # Start with G_t

            if self.use_baseline:
                # 2a. Estimate state value V(s_t, w)
                V_t = self.baseline_func.predict(tiles_t, tile_coding_indices=True)[0]
                
                # 2b. Calculate Advantage A_t = G_t - V(s_t, w)
                A_t = G - V_t 
                
                # Baseline update term is A_t (TD Error)
                td_error = A_t 
                
                # Baseline weights update: w <- w + alpha_w * td_error * x(s)
                self.baseline_func.update_weights(self.alpha_w, td_error, 
                                                  state=tiles_t, tile_coding_indices=True)
        
            # --- 2c. Policy Weight Update (theta) ---
            
            # Predict action preferences h(s, a)
            h_values = self.actor.predict(tiles_t, tile_coding_indices=True)
            
            # Ensure input to softmax is 2D (1, N)
            h_values_2d = h_values.reshape((1, -1))
            softmax_result = softmax(h_values_2d)
            softmax_probs = softmax_result.flatten()
            
            # Policy gradient update (Chapter 13.2) is proportional to:
            # [Factor] * [Gradient of ln(pi(A_t | S_t, theta))]
            # The gradient component for action 'a' is: (1_a=A_t - pi(a|S_t))
            
            # Update weights for all actions 'a'
            for a in self.actions: 
                indicator = 1.0 if a == action_t else 0.0
                
                # Policy Gradient Factor = A_t * (indicator - pi(a|s))
                grad_factor = A_t * (indicator - softmax_probs[a])
                
                # LinearRegression.update_weights applies the step size (alpha_theta)
                self.actor.update_weights(self.alpha_theta, grad_factor, 
                                          action=a, state=tiles_t, tile_coding_indices=True)

    def reset(self):
        """
        Reset the agent's internal state.

        Resets the actor (and baseline if enabled), clears the trajectory,
        and resets cached state-action pairs for a new experiment run.
        """

        self.actor.reset_weights()
        if self.use_baseline:
            self.baseline_func.reset_weights()
        self.trajectory = []
        self.prev_tiles = None
        self.prev_action = None