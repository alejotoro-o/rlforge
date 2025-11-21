import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
from copy import deepcopy
from ..base_agent import BaseAgent 


class SACAgent(BaseAgent): 
    """
    Soft Actor-Critic (SAC) for continuous action spaces.
    Refactored to build all networks internally for proper reset and management.
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 policy_net_architecture=(64, 64), # New: Architecture for Policy
                 q_net_architecture=(64, 64),      # New: Architecture for Q-Nets
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 alpha_lr=3e-4,
                 discount=0.99,         # γ
                 tau=0.005,             # Polyak averaging factor (soft update)
                 update_frequency=1,    # How often to run an update
                 buffer_size=1000000,   # Max transitions in Replay Buffer
                 mini_batch_size=256,   # Batch size for off-policy learning
                 update_start_size=256, # Minimum buffer size before starting updates
                 tanh_squash=True,
                 action_low=None,
                 action_high=None,
                 target_entropy_factor=0.9, # Target entropy = -action_dim * factor
                 device=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Store architectures for internal building/reset (Renamed)
        self.policy_net_architecture = policy_net_architecture
        self.q_net_architecture = q_net_architecture
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.discount = discount
        self.tau = tau
        self.update_frequency = update_frequency
        self.mini_batch_size = mini_batch_size
        self.update_start_size = update_start_size

        self.tanh_squash = tanh_squash
        self.action_low = action_low
        self.action_high = action_high

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize all networks and optimizers by calling the internal reset
        self.reset_nets_and_opts(
            target_entropy_factor=target_entropy_factor,
            init_weights=True # Initial call requires initialization
        )

        # --- Off-Policy Replay Buffer ---
        self.replay_buffer = deque(maxlen=buffer_size)
        self.total_steps = 0 # Total steps collected in all envs

        # Cache for previous state (N-sized tensors)
        self.prev_state = None
        self.prev_action = None
    
    # --- Network Building Helpers ---

    def _weights_init(self, m):
        """Standard Kaiming Uniform initialization for Tanh/ReLU layers."""
        if isinstance(m, nn.Linear):
            # Use Kaiming He initialization (standard for Tanh/ReLU)
            nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _create_network(self, input_dim, output_dim, architecture):
        """
        Helper function to build a standard MLP (Sequential) for either policy or Q-value.
        """
        layers = []
        current_dim = input_dim
        
        # Hidden Layers
        for hidden_size in architecture:
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.Tanh()) # As requested
            current_dim = hidden_size
            
        # Output Layer
        layers.append(nn.Linear(current_dim, output_dim))
        
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

    def _sample_action(self, mean, deterministic=False):
        """
        Samples an action from the policy (stochastic) or returns the mean (deterministic).
        Returns: action: (B, action_dim), log_prob: (B,), raw_z: (B, action_dim) 
        """
        # Unsquashed Normal distribution
        std = torch.exp(self.log_std).expand_as(mean)
        base = Normal(mean, std)
        
        # Sample z from the base distribution
        if deterministic:
            z = mean
        else:
            z = base.rsample()  # use rsample for reparameterization

        log_prob_z = base.log_prob(z).sum(dim=-1)  # (B,)

        if self.tanh_squash:
            # Tanh squash
            a = torch.tanh(z).clamp(-0.999999, 0.999999) 
            
            # Log-prob correction for tanh: sum over dims
            correction = torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1) 
            log_prob = log_prob_z - correction 
            
            # Affine rescale to [low, high] if provided
            if (self.action_low is not None) and (self.action_high is not None):
                low = self._to_tensor(self.action_low)
                high = self._to_tensor(self.action_high)
                action = 0.5 * (high + low) + 0.5 * (high - low) * a
            else:
                action = a
        else:
            action = z
            log_prob = log_prob_z

        return action, log_prob, z

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
            mean = self.policy_net(S)
            actions, _, _ = self._sample_action(mean, deterministic)

        # Cache last per-env transition (N-sized tensors)
        self.prev_state  = S
        self.prev_action = actions

        return actions.detach().cpu().numpy()


    def step_batch(self, rewards, next_states, dones, deterministic=False):
        N_envs = rewards.shape[0]
        S_prime = self._to_tensor(next_states) # S_{t+1} (N, state_dim)

        # 1. Store transitions (S_t, A_t, R_t, S_{t+1}, Done_t) into Replay Buffer
        for i in range(N_envs):
            transition = (
                self.prev_state[i].cpu().numpy(),
                self.prev_action[i].cpu().numpy(),
                rewards[i],
                next_states[i], # already numpy
                dones[i]
            )
            self.replay_buffer.append(transition)
            self.total_steps += 1


        # 2. Calculate next action A_{t+1}
        self.policy_net.eval()
        with torch.no_grad():
            mean = self.policy_net(S_prime)
            actions, _, _ = self._sample_action(mean, deterministic)

        # 3. Cache S_{t+1}, A_{t+1} for next step
        self.prev_state  = S_prime
        self.prev_action = actions

        # 4. Run SAC update if conditions are met
        if self.total_steps >= self.update_start_size and (self.total_steps % self.update_frequency == 0):
            self._sac_update()
            
        return actions.detach().cpu().numpy()


    def end_batch(self, rewards):
        N_envs = rewards.shape[0]
        
        # Store final transition (S_t, A_t, R_t, S_{t+1}=S_t, Done_t=True)
        for i in range(N_envs):
            transition = (
                self.prev_state[i].cpu().numpy(),
                self.prev_action[i].cpu().numpy(),
                rewards[i],
                self.prev_state[i].cpu().numpy(), # S_{t+1} is S_t when done
                True
            )
            self.replay_buffer.append(transition)
            self.total_steps += 1
            
        if self.total_steps >= self.update_start_size and (self.total_steps % self.update_frequency == 0):
            self._sac_update()

    
    def _sac_update(self):
        # Set all networks to training mode, including target networks for Polyak update
        self._set_device_and_train_mode(self.policy_net, True)
        self._set_device_and_train_mode(self.q_net1, True)
        self._set_device_and_train_mode(self.q_net2, True)

        if len(self.replay_buffer) < self.mini_batch_size:
            return

        # 1. Sample mini-batch from the Replay Buffer
        transitions = random.sample(self.replay_buffer, self.mini_batch_size)
        batch = list(zip(*transitions))
        
        states = self._to_tensor(np.array(batch[0])) 
        actions = self._to_tensor(np.array(batch[1])) 
        rewards = self._to_tensor(np.array(batch[2])).unsqueeze(-1)
        next_states = self._to_tensor(np.array(batch[3]))
        dones = torch.as_tensor(np.array(batch[4]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # --- Q-Network Update (Critic) ---
        with torch.no_grad():
            # Sample next action from the current policy: a' ~ π(s')
            next_mean = self.policy_net(next_states)
            next_actions, next_log_probs, _ = self._sample_action(next_mean)
            
            # Target Q-values (Double Q-learning: take min of two targets)
            # Use the helper for concatenation since Q-nets are Sequential
            q_target1 = self._q_net_forward(self.target_q_net1, next_states, next_actions)
            q_target2 = self._q_net_forward(self.target_q_net2, next_states, next_actions)
            min_q_target = torch.min(q_target1, q_target2)
            
            # Entropy-Regularized Bellman Target: Y = R + γ * (1 - D) * [min(Q') - α * logπ(a'|s')]
            next_q_target = min_q_target - self.alpha.detach() * next_log_probs.unsqueeze(-1)
            target_q = rewards + self.discount * (1 - dones) * next_q_target

        # Current Q-values
        q1_pred = self._q_net_forward(self.q_net1, states, actions)
        q2_pred = self._q_net_forward(self.q_net2, states, actions)
        
        # Q-Loss (MSE)
        q_loss1 = 0.5 * (q1_pred - target_q).pow(2).mean()
        q_loss2 = 0.5 * (q2_pred - target_q).pow(2).mean()
        
        # Optimize Q-Networks
        self.critic_opt1.zero_grad(set_to_none=True)
        q_loss1.backward()
        self.critic_opt1.step()
        
        self.critic_opt2.zero_grad(set_to_none=True)
        q_loss2.backward()
        self.critic_opt2.step()
        
        # --- Policy Network Update (Actor) ---
        # Sample action from the current policy: a ~ π(s)
        mean_s = self.policy_net(states)
        actions_reparam, log_probs, _ = self._sample_action(mean_s) 
        
        # Evaluate Q-values for the sampled actions (using the standard Q-nets)
        q1_s = self._q_net_forward(self.q_net1, states, actions_reparam)
        q2_s = self._q_net_forward(self.q_net2, states, actions_reparam)
        min_q_s = torch.min(q1_s, q2_s)
        
        # Actor Loss: Minimize E_{a~π}[-min(Q(s, a)) + α * logπ(a|s)]
        actor_loss = (self.alpha.detach() * log_probs.unsqueeze(-1) - min_q_s).mean()
        
        # Optimize Actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        # --- Temperature/Alpha Update (Auto Entropy) ---
        # Loss: L_α = E_{a~π} [ -α * (logπ(a|s) + H_target) ]
        alpha_loss = (-self.log_alpha * (log_probs.unsqueeze(-1).detach() + self.target_entropy)).mean()
        
        # Optimize Alpha
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        
        # Update self.alpha variable
        self.alpha = self.log_alpha.exp()
        
        # --- Target Network Soft Update (Polyak Averaging) ---
        with torch.no_grad():
            for param, target_param in zip(self.q_net1.parameters(), self.target_q_net1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.q_net2.parameters(), self.target_q_net2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def reset_nets_and_opts(self, target_entropy_factor=0.9, init_weights=False):
        """
        Internal function to build/rebuild all networks and optimizers.
        """
        # 1. Build/Rebuild Policy and Q Networks with fresh weights
        # Policy Net: Input = state_dim, Output = action_dim (mean)
        self.policy_net = self._create_network(
            self.state_dim, self.action_dim, self.policy_net_architecture
        ).to(self.device)
        
        # Q Nets: Input = state_dim + action_dim, Output = 1 (Q-value)
        q_input_dim = self.state_dim + self.action_dim
        self.q_net1 = self._create_network(
            q_input_dim, 1, self.q_net_architecture
        ).to(self.device)
        self.q_net2 = self._create_network(
            q_input_dim, 1, self.q_net_architecture
        ).to(self.device)


        # 2. Deep Copy for Target Networks
        self.target_q_net1 = deepcopy(self.q_net1).to(self.device)
        self.target_q_net2 = deepcopy(self.q_net2).to(self.device)
        self._set_device_and_train_mode(self.target_q_net1, False)
        self._set_device_and_train_mode(self.target_q_net2, False)
        
        # 3. Reinitialize Learnable Parameters (Log Std and Log Alpha)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.device))
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        
        # 4. Reinitialize Target Entropy 
        if init_weights:
            self.target_entropy = -float(self.action_dim) * target_entropy_factor 
            
        # 5. Reinitialize Optimizers
        # Policy optimizer includes log_std
        self.actor_opt = optim.Adam(list(self.policy_net.parameters()) + [self.log_std], lr=self.actor_lr)
        self.critic_opt1 = optim.Adam(self.q_net1.parameters(), lr=self.critic_lr)
        self.critic_opt2 = optim.Adam(self.q_net2.parameters(), lr=self.critic_lr)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        
        # 6. Update alpha
        self.alpha = self.log_alpha.exp()


    def reset(self):
        """
        Resets the agent state for a new, independent run by rebuilding all 
        networks, re-initializing optimizers, and clearing the buffer.
        """
        # Rebuild all nets, re-init optimizers, log_std, and log_alpha
        self.reset_nets_and_opts()
        
        # Reset buffers and tracking state
        self.replay_buffer.clear()
        self.total_steps = 0
        self.prev_state = None
        self.prev_action = None