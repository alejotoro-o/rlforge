import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Independent
from ..base_agent import BaseAgent # Assuming BaseAgent is available
import copy

class PPOContinuous(BaseAgent):
    """
    PPO for continuous action spaces with GAE(λ), adapted for vectorized (N) environments.
    Data is collected in (T, N, ...) format and flattened to (T*N, ...) for training.
    
    The networks are now built internally using the provided architecture to ensure 
    proper re-initialization during reset().
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 network_architecture=[64, 64], # <-- New standard argument for structure
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 discount=0.99,         # γ
                 gae_lambda=0.95,       # λ for GAE
                 clip_epsilon=0.2,
                 update_epochs=10,
                 mini_batch_size=64,
                 rollout_length=2048,   # Total number of transitions (T * N)
                 value_coef=0.5,
                 entropy_coeff=0.0,
                 max_grad_norm=0.5,
                 tanh_squash=False,
                 action_low=None,
                 action_high=None,
                 device=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network_architecture = network_architecture # Store architecture for reset

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.rollout_length = rollout_length
        self.value_coef = value_coef
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm

        self.tanh_squash = tanh_squash
        self.action_low = action_low
        self.action_high = action_high

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- FIX: Build Networks Internally ---
        self.policy_net = self._create_network(state_dim, action_dim).to(self.device)
        self.value_net  = self._create_network(state_dim, 1).to(self.device)

        # Learnable log_std (diagonal covariance)
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))
        
        # Optimizers (policy parameters + log_std)
        self.actor_opt = optim.Adam(list(self.policy_net.parameters()) + [self.log_std], lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.value_net.parameters(), lr=self.critic_lr)

        # --- Vectorized Rollout Buffer ---
        self.rollout_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'old_log_probs': [],
            'values': [],
            'dones': [],
        }
        self.step_count = 0 
        
        # Cache for previous transition (N-sized tensors)
        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None


    def _create_network(self, input_dim, output_dim):
        """
        Helper function to build a standard MLP (Sequential) for either policy or value.
        This handles the network construction internally, supporting the reset logic.
        """
        layers = []
        current_dim = input_dim
        
        # Hidden Layers
        for hidden_size in self.network_architecture:
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.Tanh()) # Use Tanh as per your original request
            current_dim = hidden_size
            
        # Output Layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Use Kaiming He initialization (standard for Tanh/ReLU)
        net = nn.Sequential(*layers)
        net.apply(self._weights_init)
        return net


    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            # Kaiming uniform initialization is standard for Tanh/ReLU layers
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='tanh') 
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        # Note: nn.Sequential and nn.Tanh don't need explicit initialization


    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _dist_from_mean(self, mean):
        # mean: (B, action_dim)
        std = torch.exp(self.log_std)           # (action_dim,)
        std = std.expand_as(mean)               # (B, action_dim)
        base = Normal(mean, std)                # elementwise normal
        return Independent(base, 1)             # treat as multivariate with diagonal cov


    def _sample_action(self, mean):
        # Unsquashed Normal
        std = torch.exp(self.log_std).expand_as(mean)
        base = Normal(mean, std)
        z = base.rsample()  # use rsample for reparameterization (optional)
        log_prob_z = base.log_prob(z).sum(dim=-1)  # (B,)

        if self.tanh_squash:
            # Tanh squash
            a = torch.tanh(z)
            # Log-prob correction for tanh: sum over dims
            correction = torch.log1p(-a.pow(2) + 1e-6).sum(dim=-1) 
            log_prob = log_prob_z - correction  # (B,)

            # Affine rescale to [low, high] if provided
            if (self.action_low is not None) and (self.action_high is not None):
                low = self._to_tensor(self.action_low)
                high = self._to_tensor(self.action_high)
                a = 0.5 * (high + low) + 0.5 * (high - low) * a
            action = a
        else:
            action = z
            log_prob = log_prob_z

        return action, log_prob


    # --- Single-env methods (wrap batch versions) ---
    def start(self, state):
        actions = self.start_batch(np.expand_dims(state, axis=0))
        return actions[0]

    def step(self, reward, state, done=False):
        actions = self.step_batch(
            np.array([reward], dtype=np.float32),
            np.expand_dims(state, axis=0),
            np.array([done], dtype=np.bool_)
        )
        return actions[0]

    def end(self, reward):
        self.end_batch(np.array([reward], dtype=np.float32))


    # --- Batch versions (true implementation) ---
    def start_batch(self, states):
        S = self._to_tensor(states)
        self.policy_net.eval()
        self.value_net.eval()
        with torch.no_grad():
            mean = self.policy_net(S)
            actions, log_probs = self._sample_action(mean)
            values = self.value_net(S).squeeze(-1)

        self.prev_state  = S
        self.prev_action = actions
        self.prev_log_prob = log_probs
        self.prev_value  = values

        return actions.detach().cpu().numpy()


    def step_batch(self, rewards, states, dones):
        # Store transition (S_t, A_t, R_t, V_t, done_t) from last step/start
        self.rollout_buffer['states'].append(self.prev_state)
        self.rollout_buffer['actions'].append(self.prev_action)
        self.rollout_buffer['rewards'].append(self._to_tensor(rewards))
        self.rollout_buffer['old_log_probs'].append(self.prev_log_prob)
        self.rollout_buffer['values'].append(self.prev_value)
        self.rollout_buffer['dones'].append(torch.as_tensor(dones, dtype=torch.bool, device=self.device))

        self.step_count += 1

        # Calculate next action A_{t+1} and V(S_{t+1})
        S = self._to_tensor(states)
        self.policy_net.eval()
        self.value_net.eval()
        with torch.no_grad():
            mean = self.policy_net(S)
            actions, log_probs = self._sample_action(mean)
            values = self.value_net(S).squeeze(-1)

        # Cache S_{t+1}, A_{t+1}, log_prob_{t+1}, V(S_{t+1}) for next step
        self.prev_state  = S
        self.prev_action = actions
        self.prev_log_prob = log_probs
        self.prev_value  = values

        if self.step_count * len(rewards) >= self.rollout_length:
            self._ppo_update()
            self.rollout_buffer = {k: [] for k in self.rollout_buffer}
            self.step_count = 0

        return actions.detach().cpu().numpy()


    def end_batch(self, rewards):
        N = len(rewards)
        
        # Store final transition (S_t, A_t, R_t, V_t, done_t=True)
        self.rollout_buffer['states'].append(self.prev_state)
        self.rollout_buffer['actions'].append(self.prev_action)
        self.rollout_buffer['rewards'].append(self._to_tensor(rewards))
        self.rollout_buffer['old_log_probs'].append(self.prev_log_prob)
        self.rollout_buffer['values'].append(self.prev_value)
        self.rollout_buffer['dones'].append(torch.ones(N, dtype=torch.bool, device=self.device))
        
        self.step_count += 1

        if self.step_count * N >= self.rollout_length:
            self._ppo_update()
            self.rollout_buffer = {k: [] for k in self.rollout_buffer}
            self.step_count = 0

    def _compute_returns_and_advantages(self, rewards, dones, values, last_value):
        T, N = rewards.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)

        next_values = torch.cat([values[1:], last_value.unsqueeze(0)], dim=0)
        next_non_terminal = (~dones).float() 

        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            delta = rewards[t] + self.discount * next_values[t] * next_non_terminal[t] - values[t]
            gae = delta + self.discount * self.gae_lambda * next_non_terminal[t] * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        return returns, advantages
    
    def _log_prob_actions(self, mean, actions):
        std = torch.exp(self.log_std).expand_as(mean)
        base = Normal(mean, std)

        if self.tanh_squash and (self.action_low is not None) and (self.action_high is not None):
            low = self._to_tensor(self.action_low)
            high = self._to_tensor(self.action_high)
            a = 2 * (actions - 0.5 * (high + low)) / (high - low).clamp_min(1e-6)
        else:
            a = actions

        if self.tanh_squash:
            a = a.clamp(-0.999999, 0.999999)
            z = 0.5 * (torch.log1p(a) - torch.log1p(-a))
            log_prob_z = base.log_prob(z).sum(dim=-1)
            correction = torch.log1p(-torch.tanh(z).pow(2) + 1e-6).sum(dim=-1)
            return log_prob_z - correction
        else:
            return base.log_prob(a).sum(dim=-1)

    def _ppo_update(self):
        self.policy_net.train()
        self.value_net.train()

        T_max = self.step_count
        N_envs = len(self.rollout_buffer['rewards'][0])

        # 1. Stack rollout buffers into (T, N, ...) tensors
        states = torch.stack(self.rollout_buffer['states'][:T_max]) 
        actions = torch.stack(self.rollout_buffer['actions'][:T_max]) 
        rewards = torch.stack(self.rollout_buffer['rewards'][:T_max]) 
        old_log_probs = torch.stack(self.rollout_buffer['old_log_probs'][:T_max])
        values = torch.stack(self.rollout_buffer['values'][:T_max]) 
        dones = torch.stack(self.rollout_buffer['dones'][:T_max]) 

        # 2. Compute GAE and returns
        last_value = self.prev_value if self.prev_value is not None else torch.zeros(N_envs, device=self.device)
        returns, advantages = self._compute_returns_and_advantages(rewards, dones, values, last_value)

        # 3. Reshape all data into flat tensors (T*N, ...) for PPO mini-batches
        T_times_N = T_max * N_envs 
        
        flat_states = states.view(T_times_N, -1)
        flat_actions = actions.view(T_times_N, -1)
        flat_old_log_probs = old_log_probs.view(T_times_N)
        flat_returns = returns.view(T_times_N)
        
        flat_advantages = advantages.view(T_times_N)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # 4. PPO Optimization Loop
        T_eff = T_times_N
        idx = torch.arange(T_eff, device=self.device)

        for _ in range(self.update_epochs):
            perm = idx[torch.randperm(T_eff)]
            for start in range(0, T_eff, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = perm[start:end]
                if batch_idx.numel() == 0:
                    continue

                batch_states = flat_states[batch_idx]
                batch_actions = flat_actions[batch_idx]
                batch_old_log_probs = flat_old_log_probs[batch_idx]
                batch_returns = flat_returns[batch_idx]
                batch_advantages = flat_advantages[batch_idx]

                # Actor forward
                mean = self.policy_net(batch_states)
                dist = self._dist_from_mean(mean)
                new_log_probs = self._log_prob_actions(mean, batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                obj1 = ratios * batch_advantages
                obj2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -(torch.min(obj1, obj2).mean() + self.entropy_coeff * entropy)

                # Critic (0.5 * MSE) scaled
                values_pred = self.value_net(batch_states).squeeze(-1)
                value_err = values_pred - batch_returns
                critic_loss = self.value_coef * 0.5 * value_err.pow(2).mean()

                # Optimize actor
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy_net.parameters()) + [self.log_std], self.max_grad_norm)
                self.actor_opt.step()

                # Optimize critic
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.critic_opt.step()

    def reset(self):
        """
        Resets the agent state for a new, independent run by rebuilding the networks
        using the stored architecture, guaranteeing fresh, random weights.
        """
        # 1. Rebuild and Re-randomize Networks using the internal builder
        self.policy_net = self._create_network(self.state_dim, self.action_dim).to(self.device)
        self.value_net  = self._create_network(self.state_dim, 1).to(self.device)
        
        # 2. Reinitialize the learnable standard deviation parameter
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.device))
        
        # 3. Reinitialize optimizers
        self.actor_opt = optim.Adam(list(self.policy_net.parameters()) + [self.log_std], lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.value_net.parameters(), lr=self.critic_lr)
        
        # Reset buffers and state
        self.rollout_buffer = {k: [] for k in self.rollout_buffer} 
        self.step_count = 0
        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None