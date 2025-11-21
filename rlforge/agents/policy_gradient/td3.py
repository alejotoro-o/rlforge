import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from copy import deepcopy
from ..base_agent import BaseAgent # Assuming BaseAgent is available

class TD3Agent(BaseAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) for continuous action spaces.
    Enhances DDPG with three core mechanisms: Twin Critics, Delayed Policy Updates,
    and Target Policy Smoothing.
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 policy_net_architecture=(256, 256),
                 q_net_architecture=(256, 256),
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 discount=0.99,          # γ
                 tau=0.005,              # Polyak averaging factor (soft update)
                 update_frequency=1,     # How often to run an update
                 buffer_size=1000000,
                 mini_batch_size=256,
                 update_start_size=256,
                 action_low=None,
                 action_high=None,
                 noise_std=0.1,          # Exploration noise std (applied to action)
                 # --- TD3 Specific Parameters ---
                 policy_delay=2,         # Policy and Target Networks update delay
                 target_noise_std=0.2,   # Std for noise added to target actions
                 target_noise_clip=0.5,  # Clipping value for target noise
                 device=None):

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Store architectures and general params
        self.policy_net_architecture = policy_net_architecture
        self.q_net_architecture = q_net_architecture
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount = discount
        self.tau = tau
        self.update_frequency = update_frequency
        self.mini_batch_size = mini_batch_size
        self.update_start_size = update_start_size
        self.noise_std = noise_std

        # --- TD3 Specific Setup ---
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip

        # Action Bounds (stored as NumPy arrays)
        if action_low is not None and not isinstance(action_low, np.ndarray):
             action_low = np.array([action_low] * action_dim)
             action_high = np.array([action_high] * action_dim)

        self.action_low = action_low
        self.action_high = action_high

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize all networks and optimizers (including twin Q-nets)
        self.reset_nets_and_opts()

        # Off-Policy Replay Buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.total_steps = 0 # Total steps collected

        # Cache for previous state/action (N-sized tensors for vectorized envs)
        self.prev_state = None
        self.prev_action = None
        self.prev_deterministic_action = None


    # --- Network Building Helpers (Inherited from DDPG) ---

    def _weights_init(self, m):
        """Standard Kaiming Uniform initialization for Tanh/ReLU layers."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _create_network(self, input_dim, output_dim, architecture, final_activation=None):
        """Helper function to build a standard MLP (Sequential)."""
        layers = []
        current_dim = input_dim

        # Hidden Layers
        for hidden_size in architecture:
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.ReLU())
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
        """Convert numpy array/python value to float32 tensor on device."""
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _q_net_forward(self, q_net, state, action):
        """Wrapper for Q-Net forward pass which requires concatenation."""
        sa = torch.cat([state, action], dim=-1)
        return q_net(sa)

    def _sample_action(self, mean, deterministic=False, action_low_np=None, action_high_np=None):
        """
        Calculates the final action, optionally adding exploration noise (for training).
        """
        low = self._to_tensor(action_low_np)
        high = self._to_tensor(action_high_np)

        # Affine rescale: [-1, 1] output (mean) -> [low, high]
        action = 0.5 * (high + low) + 0.5 * (high - low) * mean

        if not deterministic and self.noise_std > 0:
            # Add exploration noise (Gaussian)
            noise = torch.randn_like(action) * self.noise_std
            action = action + noise

        # Clip action to environment bounds after adding noise
        action = torch.clamp(action, low, high)

        return action

    # --- Standard RL Agent Interface (Wrapper Methods) ---
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

    # --- Batch implementation ---
    def start_batch(self, states, deterministic=False):
        S = self._to_tensor(states)
        self.policy_net.eval()
        with torch.no_grad():
            mean = self.policy_net(S)
            actions = self._sample_action(
                mean,
                deterministic=deterministic,
                action_low_np=self.action_low,
                action_high_np=self.action_high
            )

        self.prev_state = S
        self.prev_action = actions

        return actions.detach().cpu().numpy()

    def step_batch(self, rewards, next_states, dones, deterministic=False):
        N_envs = rewards.shape[0]
        S_prime = self._to_tensor(next_states)
        R = self._to_tensor(rewards)

        # 1. Store transitions
        for i in range(N_envs):
            transition = (
                self.prev_state[i].cpu().numpy(),
                self.prev_action[i].cpu().numpy(),
                R[i].item(),
                next_states[i],
                dones[i]
            )
            self.replay_buffer.append(transition)
            self.total_steps += 1

        # 2. Calculate next action
        self.policy_net.eval()
        with torch.no_grad():
            mean = self.policy_net(S_prime)
            actions = self._sample_action(
                mean,
                deterministic=deterministic,
                action_low_np=self.action_low,
                action_high_np=self.action_high
            )

        # 3. Cache S_{t+1}, A_{t+1}
        self.prev_state = S_prime
        self.prev_action = actions

        # 4. Run TD3 update if conditions are met
        if self.total_steps >= self.update_start_size and (self.total_steps % self.update_frequency == 0):
            self._td3_update()

        return actions.detach().cpu().numpy()

    def end_batch(self, rewards):
        N_envs = rewards.shape[0]
        R = self._to_tensor(rewards)

        # Store final transition (S_t, A_t, R_t, S_{t+1}=S_t, Done_t=True)
        for i in range(N_envs):
            transition = (
                self.prev_state[i].cpu().numpy(),
                self.prev_action[i].cpu().numpy(),
                R[i].item(),
                self.prev_state[i].cpu().numpy(),
                True
            )
            self.replay_buffer.append(transition)
            self.total_steps += 1

        if self.total_steps >= self.update_start_size and (self.total_steps % self.update_frequency == 0):
            self._td3_update()


    def _td3_update(self):
        """TD3 core update logic."""
        # Set all networks to training mode, including target networks for Polyak update
        # Policy is only updated on policy_delay cycles, but we set train mode every time for critic updates.
        self._set_device_and_train_mode(self.policy_net, True)
        self._set_device_and_train_mode(self.q_net1, True)
        self._set_device_and_train_mode(self.q_net2, True)

        if len(self.replay_buffer) < self.mini_batch_size:
            return

        # 1. Sample mini-batch
        transitions = random.sample(self.replay_buffer, self.mini_batch_size)
        batch = list(zip(*transitions))

        states = self._to_tensor(np.array(batch[0]))
        actions = self._to_tensor(np.array(batch[1]))
        rewards = self._to_tensor(np.array(batch[2])).unsqueeze(-1)
        next_states = self._to_tensor(np.array(batch[3]))
        dones = torch.as_tensor(np.array(batch[4]), dtype=torch.float32, device=self.device).unsqueeze(-1)

        # --- Critic Update (Q1 and Q2) ---
        with torch.no_grad():
            # Target Policy Smoothing: Add clipped noise to the target action a'
            # a' = π_target(s')
            next_actions_target_base = self.target_policy_net(next_states)

            # Target Policy Smoothing Noise: N(0, target_noise_std) clipped to target_noise_clip
            noise = torch.randn_like(next_actions_target_base) * self.target_noise_std
            noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
            
            # Apply noise and clip the resulting action to the environment bounds
            next_actions_target = next_actions_target_base + noise
            
            low = self._to_tensor(self.action_low)
            high = self._to_tensor(self.action_high)
            next_actions_target = torch.clamp(next_actions_target, low, high)
            
            # Twin Q-Networks: Compute Q-values using both target critics
            q_target1 = self._q_net_forward(self.target_q_net1, next_states, next_actions_target)
            q_target2 = self._q_net_forward(self.target_q_net2, next_states, next_actions_target)

            # Clipped Double Q-Learning: Use the minimum Q-value for the target (min(Q1, Q2))
            min_q_target = torch.min(q_target1, q_target2)

            # Target Q-value: Y = R + γ * (1 - D) * min(Q1_target(s', a'), Q2_target(s', a'))
            target_q = rewards + self.discount * (1 - dones) * min_q_target

        # Current Q-values (using the executed noisy action 'a')
        q1_pred = self._q_net_forward(self.q_net1, states, actions)
        q2_pred = self._q_net_forward(self.q_net2, states, actions)

        # Q-Loss (MSE for both critics)
        q1_loss = 0.5 * (q1_pred - target_q).pow(2).mean()
        q2_loss = 0.5 * (q2_pred - target_q).pow(2).mean()
        critic_loss = q1_loss + q2_loss

        # Optimize Critics
        self.critic_opt1.zero_grad(set_to_none=True)
        self.critic_opt2.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt1.step()
        self.critic_opt2.step()

        # --- Policy Update (Actor) & Target Network Soft Update ---
        # Delayed Policy Update: Only update actor and target networks every self.policy_delay steps
        if (self.total_steps // self.update_frequency) % self.policy_delay == 0:
            
            # Policy Loss: Maximize Q1(s, π(s)) -> Minimize -Q1(s, π(s))
            # Use Q1 only for the actor gradient
            actions_reparam = self.policy_net(states)
            actor_loss = -self._q_net_forward(self.q_net1, states, actions_reparam).mean()

            # Optimize Actor
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

            # Target Network Soft Update (Polyak Averaging)
            with torch.no_grad():
                # Target Q-Network 1 Update
                for param, target_param in zip(self.q_net1.parameters(), self.target_q_net1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                # Target Q-Network 2 Update
                for param, target_param in zip(self.q_net2.parameters(), self.target_q_net2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                # Target Policy Network Update
                for param, target_param in zip(self.policy_net.parameters(), self.target_policy_net.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def reset_nets_and_opts(self):
        """
        Internal function to build/rebuild all networks and optimizers,
        including the twin Q-networks.
        """
        q_input_dim = self.state_dim + self.action_dim

        # 1. Build Policy Net
        self.policy_net = self._create_network(
            self.state_dim, self.action_dim, self.policy_net_architecture, final_activation=nn.Tanh()
        ).to(self.device)

        # 2. Build Twin Q Nets (Q1 and Q2)
        self.q_net1 = self._create_network(q_input_dim, 1, self.q_net_architecture).to(self.device)
        self.q_net2 = self._create_network(q_input_dim, 1, self.q_net_architecture).to(self.device)

        # 3. Deep Copy for Target Networks
        self.target_policy_net = deepcopy(self.policy_net).to(self.device)
        self.target_q_net1 = deepcopy(self.q_net1).to(self.device)
        self.target_q_net2 = deepcopy(self.q_net2).to(self.device)
        
        self._set_device_and_train_mode(self.target_policy_net, False)
        self._set_device_and_train_mode(self.target_q_net1, False)
        self._set_device_and_train_mode(self.target_q_net2, False)

        # 4. Reinitialize Optimizers (Two for Critic)
        self.actor_opt = optim.Adam(self.policy_net.parameters(), lr=self.actor_lr)
        self.critic_opt1 = optim.Adam(self.q_net1.parameters(), lr=self.critic_lr)
        self.critic_opt2 = optim.Adam(self.q_net2.parameters(), lr=self.critic_lr)


    def reset(self):
        """
        Resets the agent state for a new, independent run.
        """
        self.reset_nets_and_opts()

        self.replay_buffer.clear()
        self.total_steps = 0
        self.prev_state = None
        self.prev_action = None