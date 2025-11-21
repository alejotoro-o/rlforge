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
    Computes the softmax over the last dimension of an array.
    Used for converting Q-values into a probability distribution for exploration.

    Args:
        x (np.ndarray): The input Q-values (typically N, action_dim).
        temperature (float): Controls the entropy of the distribution. 
                             Higher temperature results in more random actions.
    
    Returns:
        np.ndarray: The softmax probabilities (same shape as x).
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
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _create_network(self, input_dim, output_dim, architecture, final_activation=None):
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
        """Copies the weights from the main network to the target network."""
        self.target_network.load_state_dict(self.main_network.state_dict())

    def reset_networks(self):
        """Rebuilds all networks and optimizers from scratch."""
        
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
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    # --- Single-Environment API Wrappers ---
    # These mirror the DDPG structure for compatibility
    
    def start(self, state):
        actions = self.start_batch(np.expand_dims(state, axis=0)) 
        return actions[0].item()

    def step(self, reward, new_state, terminal=False):
        actions = self.step_batch(
            np.array([reward], dtype=np.float32),
            np.expand_dims(new_state, axis=0),
            np.array([terminal], dtype=np.bool_)
        )
        return actions[0].item()

    def end(self, reward):
        self.end_batch(np.array([reward], dtype=np.float32))


    # --- Vectorized Environment API (Core Implementation) ---

    def start_batch(self, states, deterministic=False):
        """
        Begin a batch of episodes by selecting actions for N environments.
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
        Take a step for N environments, update buffer with (S_t, A_t, R_t, S_{t+1}, Done_t), and train.
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
        Handle the final reward/transition for N terminated environments.
        This assumes the agent's internal prev_state/action is still valid for 
        the transitions that are now ending.
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
        Resets the agent's internal state (buffer, step counter, etc.) AND 
        rebuilds the networks via reset_networks for a fresh start.
        """
        self.reset_networks()
        self.experience_buffer.clear()
        self.elapsed_training_steps = 0
        self.total_steps = 0
        self.prev_state = None
        self.prev_action = None