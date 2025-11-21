import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from ..base_agent import BaseAgent

class PPODiscrete(BaseAgent):
    """
    PPO for discrete action spaces with GAE, adapted for vectorized (N) environments.
    - Separate actor/critic LRs
    - Rollout-based updates with GAE
    - Vectorized batch processing (T, N)
    - Advantage normalization
    - Clipped objective
    """

    def __init__(self, state_dim, 
                 num_actions,
                 actor_lr=3e-4, 
                 critic_lr=3e-4, 
                 discount=0.99,
                 clip_epsilon=0.2, 
                 network_architecture=[64, 64],
                 update_epochs=10, 
                 mini_batch_size=64,
                 rollout_length=1024, # Total transitions per env (T)
                 value_coef=0.5, 
                 entropy_coeff=0.01,
                 gae_lambda=0.95, 
                 device=None):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount = discount
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.network_architecture = network_architecture
        self.rollout_length = rollout_length
        self.value_coef = value_coef
        self.entropy_coeff = entropy_coeff
        self.gae_lambda = gae_lambda

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build actor and critic
        self.policy_net = self._build_mlp(self.state_dim, self.num_actions, self.network_architecture).to(self.device)
        self.value_net  = self._build_mlp(self.state_dim, 1,               self.network_architecture).to(self.device)

        # Separate optimizers
        self.actor_opt  = optim.Adam(self.policy_net.parameters(), lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.value_net.parameters(),  lr=self.critic_lr)

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

        # Cache for previous transition (N-sized tensors/arrays)
        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None

    def _build_mlp(self, input_dim, output_dim, hidden_layers):
        layers = []
        last = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        layers += [nn.Linear(last, output_dim)]
        return nn.Sequential(*layers)

    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    # --- Single-env methods (Wrappers) ---
    def start(self, new_state):
        """Begin a new episode with a single environment."""
        # new_state: (state_dim,) -> (1, state_dim)
        actions = self.start_batch(np.expand_dims(new_state, axis=0))
        return actions[0]   # unwrap to scalar

    def step(self, reward, new_state, done=False):
        """Take a step in a single environment."""
        actions = self.step_batch(
            np.array([reward], dtype=np.float32),
            np.expand_dims(new_state, axis=0),
            np.array([done], dtype=np.bool_)
        )
        return actions[0]   # unwrap to scalar

    def end(self, reward):
        """Complete an episode in a single environment."""
        self.end_batch(np.array([reward], dtype=np.float32))


    # --- Batch versions (Vectorized Implementation) ---
    def start_batch(self, states):
        """
        Begin a new episode with multiple environments (N).
        states: (N, state_dim)
        Returns: (N,) numpy array of actions
        """
        S = self._to_tensor(states)  # (N, state_dim)
        
        self.policy_net.eval()
        self.value_net.eval()
        
        with torch.no_grad():
            logits = self.policy_net(S)               # (N, num_actions)
            dist = Categorical(logits=logits)
            actions = dist.sample()                   # (N,)
            log_probs = dist.log_prob(actions)        # (N,)
            values = self.value_net(S).squeeze(-1)    # (N,)

        # Cache last per-env transition
        self.prev_state  = S
        self.prev_action = actions
        self.prev_log_prob = log_probs
        self.prev_value  = values

        return actions.detach().cpu().numpy()

    def step_batch(self, rewards, states, dones):
        """
        Take a step with multiple environments (N).
        rewards: (N,), states: (N, state_dim), dones: (N,)
        Returns: (N,) numpy array of actions
        """
        # Store transition (S_t, A_t, R_t, V_t, done_t) from last step/start
        self.rollout_buffer['states'].append(self.prev_state)
        self.rollout_buffer['actions'].append(self.prev_action)
        self.rollout_buffer['rewards'].append(self._to_tensor(rewards))
        self.rollout_buffer['old_log_probs'].append(self.prev_log_prob)
        self.rollout_buffer['values'].append(self.prev_value)
        self.rollout_buffer['dones'].append(torch.as_tensor(dones, dtype=torch.bool, device=self.device))

        self.step_count += 1

        # Calculate next action A_{t+1} and V(S_{t+1})
        S = self._to_tensor(states)  # (N, state_dim)
        
        self.policy_net.eval()
        self.value_net.eval()
        
        with torch.no_grad():
            logits = self.policy_net(S)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            values = self.value_net(S).squeeze(-1)

        # Cache S_{t+1} data
        self.prev_state  = S
        self.prev_action = actions
        self.prev_log_prob = log_probs
        self.prev_value  = values

        # Check update condition
        if self.step_count * len(rewards) >= self.rollout_length:
            self._ppo_update()
            self.rollout_buffer = {k: [] for k in self.rollout_buffer}
            self.step_count = 0

        return actions.detach().cpu().numpy()

    def end_batch(self, rewards):
        """
        Complete episodes for multiple environments (N).
        rewards: (N,)
        """
        N = len(rewards)
        
        # Store final transition
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

    def _compute_gae_advantages(self, rewards, dones, values, last_value):
        """
        Calculates GAE advantages and returns for vectorized (T, N) tensors.
        
        Inputs: rewards (T, N), dones (T, N), values (T, N)
        last_value: (N,) - V(S_{T+1}) or zero
        Outputs: advantages (T, N), returns (T, N)
        """
        T, N = rewards.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # Construct V(S_{t+1}) for the entire batch
        # It's values[t+1] for t < T-1, and last_value for t = T-1
        next_values = torch.cat([values[1:], last_value.unsqueeze(0)], dim=0) # (T, N)
        
        # Mask for non-terminal next states
        next_non_terminal = (~dones).float() 

        gae = torch.zeros(N, device=self.device)
        
        for t in reversed(range(T)):
            # delta_t = r_t + gamma * V(s_{t+1}) * (1-d_t) - V(s_t)
            delta = rewards[t] + self.discount * next_values[t] * next_non_terminal[t] - values[t]
            
            # A_t = delta_t + gamma * lambda * (1-d_t) * A_{t+1}
            gae = delta + self.discount * self.gae_lambda * next_non_terminal[t] * gae
            advantages[t] = gae
            
            # Return_t = A_t + V_t
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns


    def _ppo_update(self):
        self.policy_net.train()
        self.value_net.train()

        T_max = self.step_count
        N_envs = len(self.rollout_buffer['rewards'][0])

        # 1. Stack buffers -> (T, N, ...)
        states = torch.stack(self.rollout_buffer['states'][:T_max])           # (T, N, state_dim)
        actions = torch.stack(self.rollout_buffer['actions'][:T_max])         # (T, N)
        rewards = torch.stack(self.rollout_buffer['rewards'][:T_max])         # (T, N)
        old_log_probs = torch.stack(self.rollout_buffer['old_log_probs'][:T_max]) # (T, N)
        values = torch.stack(self.rollout_buffer['values'][:T_max])           # (T, N)
        dones = torch.stack(self.rollout_buffer['dones'][:T_max])             # (T, N)

        # 2. GAE Calculation
        # last_value is used for bootstrapping if the trajectory is cut off by the buffer limit
        last_value = self.prev_value if self.prev_value is not None else torch.zeros(N_envs, device=self.device)
        
        advantages, returns = self._compute_gae_advantages(rewards, dones, values, last_value)

        # 3. Flatten to (T*N, ...)
        T_times_N = T_max * N_envs
        
        flat_states = states.view(T_times_N, -1)       # (T*N, state_dim)
        flat_actions = actions.view(T_times_N)         # (T*N,)
        flat_old_log_probs = old_log_probs.view(T_times_N) # (T*N,)
        flat_returns = returns.view(T_times_N)         # (T*N,)
        
        flat_advantages = advantages.view(T_times_N)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # 4. Optimization
        T_eff = T_times_N
        idx = torch.arange(T_eff, device=self.device)

        for _ in range(self.update_epochs):
            perm = idx[torch.randperm(T_eff)]
            for start in range(0, T_eff, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = perm[start:end]
                if batch_idx.numel() == 0: continue

                batch_states = flat_states[batch_idx]       # (B, state_dim)
                batch_actions = flat_actions[batch_idx]     # (B,)
                batch_old_log_probs = flat_old_log_probs[batch_idx] # (B,)
                batch_returns = flat_returns[batch_idx]     # (B,)
                batch_advantages = flat_advantages[batch_idx] # (B,)

                # Forward
                logits = self.policy_net(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Ratios
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped Loss
                obj1 = ratios * batch_advantages
                obj2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -(torch.min(obj1, obj2).mean() + self.entropy_coeff * entropy)

                # Critic Loss
                values_pred = self.value_net(batch_states).squeeze(-1)
                value_err = values_pred - batch_returns
                critic_loss = self.value_coef * 0.5 * (value_err.pow(2).mean())

                # Optimize
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
                self.actor_opt.step()

                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
                self.critic_opt.step()

    def reset(self):
        # Reinitialize networks and optimizers
        self.policy_net = self._build_mlp(self.state_dim, self.num_actions, self.network_architecture).to(self.device)
        self.value_net  = self._build_mlp(self.state_dim, 1,               self.network_architecture).to(self.device)
        self.actor_opt  = optim.Adam(self.policy_net.parameters(), lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.value_net.parameters(),  lr=self.critic_lr)
        
        self.rollout_buffer = {k: [] for k in self.rollout_buffer}
        self.step_count = 0
        
        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None