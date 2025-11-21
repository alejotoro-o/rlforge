import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from copy import deepcopy
from ..base_agent import BaseAgent # Assuming BaseAgent is available

# Note: DDPG does not use stochastic policy, so no torch.distributions needed.

class DDPGAgent(BaseAgent): 
    """
    Deep Deterministic Policy Gradient (DDPG) for continuous action spaces.
    Adapted for compatibility with vectorized environments and internal network management.
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 policy_net_architecture=(256, 256), # Often needs larger nets than SAC/PPO
                 q_net_architecture=(256, 256),     
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 discount=0.99,          # γ
                 tau=0.001,              # Polyak averaging factor (soft update)
                 update_frequency=1,     # How often to run an update
                 buffer_size=1000000,    # Max transitions in Replay Buffer
                 mini_batch_size=64,     # Batch size for off-policy learning
                 update_start_size=256,  # Minimum buffer size before starting updates
                 action_low=None,
                 action_high=None,
                 noise_std=0.1,          # Standard deviation for Ornstein-Uhlenbeck or Gaussian noise
                 device=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Store architectures for internal building/reset
        self.policy_net_architecture = policy_net_architecture
        self.q_net_architecture = q_net_architecture
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount = discount
        self.tau = tau
        self.update_frequency = update_frequency
        self.mini_batch_size = mini_batch_size
        self.update_start_size = update_start_size

        # --- FIX: Store action bounds as NUMPY arrays or Python floats ---
        # The environment returns action bounds as (action_dim,) arrays, so we store the array
        if action_low is not None and not isinstance(action_low, np.ndarray):
             # Ensure bounds are array-like if they represent a vector action space
             action_low = np.array([action_low] * action_dim) 
             action_high = np.array([action_high] * action_dim)

        self.action_low = action_low
        self.action_high = action_high
        self.noise_std = noise_std
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize all networks and optimizers by calling the internal reset
        self.reset_nets_and_opts(init_weights=True)

        # --- Off-Policy Replay Buffer ---
        self.replay_buffer = deque(maxlen=buffer_size)
        self.total_steps = 0 # Total steps collected in all envs

        # Cache for previous state (N-sized tensors)
        self.prev_state = None
        self.prev_action = None
        
        # Cache for previous deterministic action (A_t) for noise injection
        self.prev_deterministic_action = None 
        # Stores the current OU-noise state (N, action_dim) for vectorized envs
        self.ou_noise_state = None 

    # --- Network Building Helpers (Reused from SAC) ---

    def _weights_init(self, m):
        """Standard Kaiming Uniform initialization for Tanh/ReLU layers."""
        if isinstance(m, nn.Linear):
            # Use Kaiming He initialization (standard for Tanh/ReLU)
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # DDPG often uses ReLU
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _create_network(self, input_dim, output_dim, architecture, final_activation=None):
        """
        Helper function to build a standard MLP (Sequential) for either policy or Q-value.
        Note: DDPG typically uses ReLU for hidden layers.
        """
        layers = []
        current_dim = input_dim
        
        # Hidden Layers
        for hidden_size in architecture:
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.ReLU()) # DDPG typically uses ReLU
            current_dim = hidden_size
            
        # Output Layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        if final_activation is not None:
            layers.append(final_activation)
            
        net = nn.Sequential(*layers)
        net.apply(self._weights_init)
        return net

    def _set_device_and_train_mode(self, net, requires_grad):
        """Helper to move network to device and set parameter requirement."""
        net.to(self.device)
        net.train() if requires_grad else net.eval()
        for param in net.parameters():
            param.requires_grad = requires_grad

    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)
    
    def _q_net_forward(self, q_net, state, action):
        """Wrapper for Q-Net forward pass which requires concatenation."""
        sa = torch.cat([state, action], dim=-1)
        return q_net(sa)
    
    def _sample_action(self, mean, deterministic=False, action_low_np=None, action_high_np=None):
        """
        DDPG samples a deterministic action (mean) and adds exploration noise.
        
        action_low_np/action_high_np are the action bounds (NumPy arrays).
        """
        
        # Affine rescale to [low, high] if provided
        if (action_low_np is not None) and (action_high_np is not None):
            low = self._to_tensor(action_low_np)
            high = self._to_tensor(action_high_np)
            
            # DDPG typically uses Tanh (applied in policy_net) to bound network output to [-1, 1],
            # then rescales it to the environment action space [low, high].
            action = 0.5 * (high + low) + 0.5 * (high - low) * mean
        else:
            action = mean

        if not deterministic and self.noise_std > 0:
            # Add Gaussian noise (simpler than OU noise for general use)
            noise = torch.randn_like(action) * self.noise_std
            action = action + noise
            
        # Clip action to environment bounds after adding noise
        if (action_low_np is not None) and (action_high_np is not None):
            # FIX: Use positional arguments (min, max) when passing tensors to clamp
            action = torch.clamp(action, low, high) 
        
        return action

    # --- Single-env methods (wrap batch versions) ---
    def start(self, state, deterministic=False):
        actions = self.start_batch(np.expand_dims(state, axis=0), deterministic)
        return actions[0]

    def step(self, reward, state, done=False, deterministic=False):
        actions = self.step_batch(
            np.array([reward], dtype=np.float32),
            np.expand_dims(state, axis=0),
            np.array([done], dtype=np.bool_),
            deterministic
        )
        return actions[0]

    def end(self, reward):
        self.end_batch(np.array([reward], dtype=np.float32))


    # --- Batch versions (true implementation) ---
    def start_batch(self, states, deterministic=False):
        S = self._to_tensor(states)  # (N, state_dim)
        
        self.policy_net.eval()
        with torch.no_grad():
            # Get the deterministic policy output (mean)
            mean = self.policy_net(S)
            
            # Add noise for exploration (if not deterministic)
            actions = self._sample_action(
                mean, 
                deterministic=deterministic,
                action_low_np=self.action_low,
                action_high_np=self.action_high
            )

        # Cache last per-env transition (N-sized tensors)
        self.prev_state = S
        self.prev_deterministic_action = mean # Store deterministic action before noise
        self.prev_action = actions          # Store noisy action used in env

        return actions.detach().cpu().numpy()


    def step_batch(self, rewards, next_states, dones, deterministic=False):
        N_envs = rewards.shape[0]
        S_prime = self._to_tensor(next_states) # S_{t+1} (N, state_dim)
        R = self._to_tensor(rewards)

        # 1. Store transitions (S_t, A_t, R_t, S_{t+1}, Done_t) into Replay Buffer
        for i in range(N_envs):
            # DDPG stores the noisy action (self.prev_action) that was executed
            transition = (
                self.prev_state[i].cpu().numpy(),
                self.prev_action[i].cpu().numpy(),
                R[i].item(),
                next_states[i], # already numpy
                dones[i]
            )
            self.replay_buffer.append(transition)
            self.total_steps += 1


        # 2. Calculate next action A_{t+1} (DDPG is deterministic policy + noise)
        self.policy_net.eval()
        with torch.no_grad():
            mean = self.policy_net(S_prime)
            actions = self._sample_action(
                mean, 
                deterministic=deterministic,
                action_low_np=self.action_low,
                action_high_np=self.action_high
            )

        # 3. Cache S_{t+1}, A_{t+1} for next step
        self.prev_state = S_prime
        self.prev_deterministic_action = mean 
        self.prev_action = actions

        # 4. Run DDPG update if conditions are met
        if self.total_steps >= self.update_start_size and (self.total_steps % self.update_frequency == 0):
            self._ddpg_update()
            
        return actions.detach().cpu().numpy()


    def end_batch(self, rewards):
        N_envs = rewards.shape[0]
        R = self._to_tensor(rewards)
        
        # Store final transition (S_t, A_t, R_t, S_{t+1}=S_t, Done_t=True)
        for i in range(N_envs):
            # DDPG stores the noisy action (self.prev_action) that was executed
            transition = (
                self.prev_state[i].cpu().numpy(),
                self.prev_action[i].cpu().numpy(),
                R[i].item(),
                self.prev_state[i].cpu().numpy(), # S_{t+1} is S_t when done
                True
            )
            self.replay_buffer.append(transition)
            self.total_steps += 1
            
        if self.total_steps >= self.update_start_size and (self.total_steps % self.update_frequency == 0):
            self._ddpg_update()

    
    def _ddpg_update(self):
        # Set all networks to training mode, including target networks for Polyak update
        self._set_device_and_train_mode(self.policy_net, True)
        self._set_device_and_train_mode(self.q_net, True)

        if len(self.replay_buffer) < self.mini_batch_size:
            return

        # 1. Sample mini-batch from the Replay Buffer
        transitions = random.sample(self.replay_buffer, self.mini_batch_size)
        batch = list(zip(*transitions))
        
        states = self._to_tensor(np.array(batch[0])) 
        actions = self._to_tensor(np.array(batch[1])) # This is the NOISY action used by the environment
        rewards = self._to_tensor(np.array(batch[2])).unsqueeze(-1)
        next_states = self._to_tensor(np.array(batch[3]))
        dones = torch.as_tensor(np.array(batch[4]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # --- Q-Network Update (Critic) ---
        with torch.no_grad():
            # Get the deterministic next action from the TARGET policy network: a' = π_target(s')
            next_actions_target = self.target_policy_net(next_states) 
            
            # Calculate Target Q-value: Y = R + γ * (1 - D) * Q_target(s', a')
            # The target action is deterministic and comes from the target policy network
            q_target = self._q_net_forward(self.target_q_net, next_states, next_actions_target)
            target_q = rewards + self.discount * (1 - dones) * q_target

        # Current Q-values: Q(s, a) using the *executed* noisy action 'a'
        q_pred = self._q_net_forward(self.q_net, states, actions)
        
        # Q-Loss (MSE)
        q_loss = 0.5 * (q_pred - target_q).pow(2).mean()
        
        # Optimize Q-Network
        self.critic_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.critic_opt.step()
        
        # --- Policy Network Update (Actor) ---
        # The policy is deterministic, so we maximize Q(s, π(s)) by minimizing -Q(s, π(s))
        # Need to re-enable gradients for the policy network, so we compute the actions again.
        
        # Get action from CURRENT policy network: a = π(s)
        actions_reparam = self.policy_net(states) 
        
        # Evaluate Q-values for the policy's action (using the CURRENT Q-net)
        # Note: We detach the Q-network calculation from the policy gradient flow
        actor_loss = -self._q_net_forward(self.q_net, states, actions_reparam).mean()
        
        # Optimize Actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        # --- Target Network Soft Update (Polyak Averaging) ---
        with torch.no_grad():
            # Target Q-Network Update
            for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Target Policy Network Update
            for param, target_param in zip(self.policy_net.parameters(), self.target_policy_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def reset_nets_and_opts(self, init_weights=False):
        """
        Internal function to build/rebuild all networks and optimizers.
        """
        # 1. Build/Rebuild Policy and Q Networks with fresh weights
        # Policy Net: Input = state_dim, Output = action_dim (mean), Final Activation: Tanh 
        # Tanh activation bounds the output between [-1, 1] before scaling to the env bounds.
        self.policy_net = self._create_network(
            self.state_dim, self.action_dim, self.policy_net_architecture, final_activation=nn.Tanh()
        ).to(self.device)
        
        # Q Net: Input = state_dim + action_dim, Output = 1 (Q-value), No final activation
        q_input_dim = self.state_dim + self.action_dim
        self.q_net = self._create_network(
            q_input_dim, 1, self.q_net_architecture
        ).to(self.device)


        # 2. Deep Copy for Target Networks
        self.target_policy_net = deepcopy(self.policy_net).to(self.device)
        self.target_q_net = deepcopy(self.q_net).to(self.device)
        self._set_device_and_train_mode(self.target_policy_net, False)
        self._set_device_and_train_mode(self.target_q_net, False)
        
        # 3. Reinitialize Optimizers
        self.actor_opt = optim.Adam(self.policy_net.parameters(), lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.q_net.parameters(), lr=self.critic_lr)


    def reset(self):
        """
        Resets the agent state for a new, independent run by rebuilding all 
        networks, re-initializing optimizers, and clearing the buffer.
        """
        # Rebuild all nets, re-init optimizers
        self.reset_nets_and_opts()
        
        # Reset buffers and tracking state
        self.replay_buffer.clear()
        self.total_steps = 0
        self.prev_state = None
        self.prev_action = None
        self.prev_deterministic_action = None