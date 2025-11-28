import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random
from collections import deque
from ..base_agent import BaseAgent


def softmax(x, temperature=1.0):
    """
    Compute the softmax over the last dimension of an array.

    This is typically used to convert Q-values into a probability distribution
    for exploration.

    Parameters
    ----------
    x : np.ndarray
        Input Q-values, typically of shape (N, action_dim).
    temperature : float
        Controls the entropy of the distribution. Higher temperature results
        in more random actions.

    Returns
    -------
    np.ndarray
        Softmax probabilities with the same shape as `x`.
    """

    # Apply temperature
    x_temp = x / temperature
    
    # Numerically stable softmax: subtract max for exponentiation
    e_x = np.exp(x_temp - np.max(x_temp, axis=-1, keepdims=True))
    
    # Calculate softmax
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class ReplayBuffer:
    """
    An optimized fixed-size replay buffer using collections.deque for O(1) appends and pops.
    """

    def __init__(self, size, mini_batch_size):
        self.buffer = deque(maxlen=size)
        self.mini_batch_size = mini_batch_size
        self.size = size 

    def __len__(self):
        """Returns the current number of experiences stored."""
        return len(self.buffer)

    def append(self, state, action, reward, terminal, new_state):
        """Add a single new experience to the buffer."""
        self.buffer.append([state, action, reward, terminal, new_state])

    def sample(self):
        """
        Randomly sample a mini-batch of experiences from the buffer.
        """
        if len(self.buffer) < self.mini_batch_size:
            return []
            
        sampled_batch = random.sample(self.buffer, self.mini_batch_size)
        return sampled_batch
        
    def clear(self):
        """Clears the buffer content."""
        self.buffer.clear()


# --- Main Agent Class ---

class DQNTorchAgent(BaseAgent):
    """
    Deep Q-Network (DQN) Agent implemented in PyTorch.

    This agent uses a feedforward neural network to approximate Q-values
    for discrete actions. It supports both single-environment and
    vectorized-environment APIs, experience replay, and a target network
    for stable training.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state space.
    action_dim : int
        Number of discrete actions available in the environment.
    network_architecture : tuple of int, optional
        Sizes of hidden layers in the Q-network (default=(64, 64)).
    learning_rate : float, optional
        Learning rate for the optimizer (default=1e-3).
    discount : float, optional
        Discount factor γ applied to future rewards (default=0.99).
    temperature : float, optional
        Temperature parameter for softmax exploration (default=1.0).
    target_network_update_steps : int, optional
        Number of training steps between target network synchronizations (default=1000).
    num_replay : int, optional
        Number of replay updates per environment step (default=1).
    experience_buffer_size : int, optional
        Maximum size of the replay buffer (default=100000).
    mini_batch_size : int, optional
        Size of mini-batches sampled from the replay buffer (default=32).
    device : str or torch.device, optional
        Device to run computations on ("cpu" or "cuda").
    """

    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 network_architecture=(64, 64),
                 learning_rate=1e-3, 
                 discount=0.99,
                 temperature=1.0, 
                 target_network_update_steps=1000,
                 num_replay=1, 
                 experience_buffer_size=100000,
                 mini_batch_size=32, 
                 device="cpu"):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network_architecture = network_architecture
        self.discount = discount
        self.temperature = temperature
        self.target_network_update_steps = target_network_update_steps
        self.num_replay = num_replay
        self.mini_batch_size = mini_batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._initial_lr = learning_rate

        self.main_network = None
        self.target_network = None
        self.optimizer = None
        self.reset_networks() # Build initial networks and optimizers

        self.experience_buffer = ReplayBuffer(experience_buffer_size, mini_batch_size)
        self.elapsed_training_steps = 0
        self.total_steps = 0

        # Cache for previous state/action (N-sized arrays/tensors for the *last* step)
        self.prev_state = None 
        self.prev_action = None 
        self.loss_fn = nn.MSELoss()
    

    # --- Network Building Helpers ---

    def _weights_init(self, m):
        """
        Initialize weights of linear layers.

        Uses Kaiming uniform initialization for weights and sets biases to zero.

        Parameters
        ----------
        m : nn.Module
            A PyTorch module (typically nn.Linear) to initialize.
        """

        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _create_network(self, input_dim, output_dim, architecture, final_activation=None):
        """
        Build a feedforward neural network.

        Constructs a sequential model with ReLU activations and optional
        final activation.

        Parameters
        ----------
        input_dim : int
            Dimension of the input features.
        output_dim : int
            Dimension of the output (number of actions).
        architecture : tuple of int
            Sizes of hidden layers.
        final_activation : nn.Module, optional
            Activation function applied to the final layer.

        Returns
        -------
        nn.Sequential
            The constructed PyTorch network.
        """

        layers = []
        current_dim = input_dim
        
        for hidden_size in architecture:
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.ReLU())
            current_dim = hidden_size
            
        layers.append(nn.Linear(current_dim, output_dim))
        
        if final_activation is not None:
            layers.append(final_activation)
            
        net = nn.Sequential(*layers)
        net.apply(self._weights_init)
        return net.to(self.device)

    def _sync_target_network(self):
        """
        Synchronize target network with main network.

        Copies the weights from the main Q-network to the target Q-network.
        """

        self.target_network.load_state_dict(self.main_network.state_dict())

    def reset_networks(self):
        """
        Reset and rebuild networks and optimizer.

        Creates a new main network and target network, initializes weights,
        and sets up the Adam optimizer.
        """
        
        self.main_network = self._create_network(
            self.state_dim, 
            self.action_dim, 
            self.network_architecture
        ).to(self.device)

        self.target_network = deepcopy(self.main_network).to(self.device)
        self.target_network.eval()
        self._sync_target_network()
        
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self._initial_lr)

    def _to_tensor(self, x):
        """
        Convert input to a PyTorch tensor.

        Parameters
        ----------
        x : array-like
            Input data.

        Returns
        -------
        torch.Tensor
            Float tensor on the agent's device.
        """

        return torch.as_tensor(x, dtype=torch.float32, device=self.device)
    
    def start(self, state):
        """
        Begin a new episode in a single environment.

        Selects an initial action based on the current state.

        Parameters
        ----------
        state : array-like
            Initial state of the environment.

        Returns
        -------
        int
            Selected action.
        """

        actions = self.start_batch(np.expand_dims(state, axis=0)) 
        return actions[0].item()

    def step(self, reward, new_state, terminal=False):
        """
        Take a step in a single environment.

        Updates replay buffer and selects the next action.

        Parameters
        ----------
        reward : float
            Reward received from the previous action.
        new_state : array-like
            Next state observed.
        terminal : bool, optional
            Whether the episode has terminated (default=False).

        Returns
        -------
        int
            Selected action.
        """

        actions = self.step_batch(
            np.array([reward], dtype=np.float32),
            np.expand_dims(new_state, axis=0),
            np.array([terminal], dtype=np.bool_)
        )
        return actions[0].item()

    def end(self, reward):
        """
        Complete an episode in a single environment.

        Stores the final transition into the replay buffer.

        Parameters
        ----------
        reward : float
            Final reward received at the end of the episode.
        """

        self.end_batch(np.array([reward], dtype=np.float32))


    # --- Vectorized Environment API (Core Implementation) ---

    def start_batch(self, states, deterministic=False):
        """
        Begin a batch of episodes.

        Selects actions for multiple environments simultaneously.

        Parameters
        ----------
        states : array-like, shape (N, state_dim)
            Batch of initial states.
        deterministic : bool, optional
            If True, selects greedy actions; otherwise uses softmax exploration.

        Returns
        -------
        numpy.ndarray
            Array of selected actions of shape (N,).
        """

        S = self._to_tensor(states) # (N, state_dim)
        
        self.main_network.eval()
        with torch.no_grad():
            q_values = self.main_network(S).cpu().numpy()
        self.main_network.train()

        if deterministic or self.temperature <= 0:
            actions = np.argmax(q_values, axis=1)
        else:
            probs = softmax(q_values, self.temperature) 
            probs = probs.reshape(-1, self.action_dim)
            # Select action based on softmax probabilities
            actions = np.array([np.random.choice(self.action_dim, p=p) for p in probs], dtype=np.int64)

        # Cache S_t, A_t for the next step (step_batch)
        self.prev_state = states # Keep as numpy array
        self.prev_action = actions # Keep as numpy array
        return actions

    def step_batch(self, rewards, next_states, dones, deterministic=False):
        """
        Take a step in multiple environments.

        Stores transitions in the replay buffer, performs training updates,
        synchronizes the target network if needed, and selects next actions.

        Parameters
        ----------
        rewards : array-like, shape (N,)
            Rewards from the previous actions.
        next_states : array-like, shape (N, state_dim)
            Next states observed.
        dones : array-like, shape (N,)
            Boolean flags indicating episode termination.
        deterministic : bool, optional
            If True, selects greedy actions; otherwise uses softmax exploration.

        Returns
        -------
        numpy.ndarray
            Array of selected actions of shape (N,).
        """

        N_envs = rewards.shape[0]

        # 1. Store transitions (S_t, A_t, R_t, S_{t+1}, Done_t) into Replay Buffer
        # Use previous state/action and current reward/next_state/done
        if self.prev_state is not None and self.prev_action is not None:
            for i in range(N_envs):
                self.experience_buffer.append(
                    self.prev_state[i], 
                    self.prev_action[i].item(),
                    rewards[i].item(),
                    dones[i].item(),
                    next_states[i]
                )
            self.total_steps += N_envs
        else:
            print("Warning: step_batch called before start_batch/initial state cache is empty.")

        # 2. Select Next Actions A_{t+1} (Inference)
        S_prime = self._to_tensor(next_states)
        self.main_network.eval()
        with torch.no_grad():
            q_values = self.main_network(S_prime).cpu().numpy()
        self.main_network.train()
            
        if deterministic or self.temperature <= 0:
            actions = np.argmax(q_values, axis=1)
        else:
            probs = softmax(q_values, self.temperature)
            probs = probs.reshape(-1, self.action_dim)
            actions = np.array([np.random.choice(self.action_dim, p=p) for p in probs], dtype=np.int64)


        # 3. Perform Training Steps and update target network
        if len(self.experience_buffer) >= self.mini_batch_size:
            for _ in range(self.num_replay):
                self._train_step()
                
            self.elapsed_training_steps += 1
            if self.elapsed_training_steps >= self.target_network_update_steps:
                self._sync_target_network()
                self.elapsed_training_steps = 0

        # 4. Update Agent State Cache for Next Step
        self.prev_state = next_states
        self.prev_action = actions
        return actions

    def end_batch(self, rewards):
        """
        Handle the final reward and transition for a batch of terminated environments.

        This method stores terminal transitions into the replay buffer and performs
        training updates if enough samples are available. It assumes that the agent's
        internal caches for previous states and actions are valid for the terminated
        episodes.

        Parameters
        ----------
        rewards : array-like, shape (N,)
            Final rewards received for each of the N terminated environments.

        Notes
        -----
        - Each terminal transition is stored as (S_t, A_t, R_t, S_{t+1}=S_t, Done=True).
        - Training is triggered after storing transitions if the replay buffer
          contains at least ``mini_batch_size`` samples.
        - The internal state/action cache is reset if all environments in the batch
          have terminated.

        """

        N_envs = rewards.shape[0]
        R = np.atleast_1d(rewards) 
        
        # Guard against calling end_batch when prev_state/action is None
        if self.prev_state is None or self.prev_action is None:
             print("Warning: end_batch called but prev_state/action cache is empty. Ignoring transition.")
             return

        # Store final transition (S_t, A_t, R_t, S_{t+1}=S_t, Done_t=True)
        # We assume the N_envs rewards correspond to the first N_envs entries 
        # in the prev_state/action caches for the terminated episodes.
        for i in range(N_envs):
            # Final state is stored as S_t (self.prev_state[i])
            self.experience_buffer.append(
                self.prev_state[i], 
                self.prev_action[i].item(),
                R[i].item(),
                True, # Terminal
                self.prev_state[i] # S_{t+1} = S_t for terminal transition
            )
            self.total_steps += 1
            
        # Perform training after storing the final transition
        if len(self.experience_buffer) >= self.mini_batch_size:
            for _ in range(self.num_replay):
                self._train_step()
                
            self.elapsed_training_steps += 1
            if self.elapsed_training_steps >= self.target_network_update_steps:
                self._sync_target_network()
                self.elapsed_training_steps = 0
                
        # IMPORTANT: Since this is an episodic batch, we only reset the cache 
        # if the runner tells us the entire batch has ended (i.e., N_envs == total envs).
        # For a simple N=1 setup, this means if we got 1 reward, the episode is over.
        # Since the ExperimentRunner handles resetting the env, we keep the cache simple.
        if N_envs == self.prev_state.shape[0]:
            self.prev_state = None
            self.prev_action = None


    def _train_step(self):
        """
        Perform a single training step using a mini-batch from the replay buffer.

        Samples a batch of transitions, computes the TD target using the target
        network, and updates the main Q-network by minimizing the mean squared
        error between predicted Q-values and targets.

        Workflow
        --------
        1. Sample a mini-batch of transitions from the replay buffer.
        2. Compute Q(s,a) for the sampled states and actions using the main network.
        3. Compute TD targets: ``y = r + γ * max(Q_target(s')) * (1 - terminal)``.
        4. Calculate loss as MSE between Q(s,a) and targets.
        5. Backpropagate and update network parameters with gradient clipping.

        Returns
        -------
        None
            Updates the main network weights in-place.
        """

        sampled_batch = self.experience_buffer.sample()
        if not sampled_batch:
            return 
            
        states, actions, rewards, terminal, new_states = map(list, zip(*sampled_batch))

        # Data preparation (Tensors of shape (B, ...))
        states = self._to_tensor(np.vstack(states))
        actions = torch.tensor(np.vstack(actions).squeeze(), dtype=torch.int64).to(self.device).unsqueeze(1)
        rewards = torch.tensor(np.vstack(rewards).squeeze(), dtype=torch.float32).to(self.device)
        terminal = torch.tensor(np.vstack(terminal).squeeze(), dtype=torch.float32).to(self.device)
        new_states = self._to_tensor(np.vstack(new_states))

        # --- Calculate Q(s,a) (Main Network) ---
        q_values = self.main_network(states)
        # Use actions to index Q-values: Q(s,a)
        q_values_vec = q_values.gather(1, actions).squeeze()

        # --- Calculate Target y = r + gamma * max(Q_target(s')) ---
        with torch.no_grad():
            target_q_values = self.target_network(new_states)
            # Find the maximum Q-value for the next state
            max_next_q = target_q_values.max(1)[0]
            # TD Target: r + gamma * max(Q_target(s')) * (1 - terminal)
            target = rewards + self.discount * max_next_q * (1.0 - terminal)

        # Compute loss (TD Error)
        loss = self.loss_fn(q_values_vec, target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=0.5)
        self.optimizer.step()

    def reset(self):
        """
        Reset the agent's internal state and networks.

        Clears the replay buffer, resets counters, and rebuilds the main and target
        networks for a fresh start.

        Notes
        -----
        - Resets ``elapsed_training_steps`` and ``total_steps`` to zero.
        - Clears cached previous state and action.
        - Calls ``reset_networks()`` to reinitialize the Q-networks and optimizer.
        """

        self.reset_networks()
        self.experience_buffer.clear()
        self.elapsed_training_steps = 0
        self.total_steps = 0
        self.prev_state = None
        self.prev_action = None