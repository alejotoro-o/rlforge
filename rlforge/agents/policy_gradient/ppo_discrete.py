import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# If you have a BaseAgent in rlforge, you can inherit; otherwise remove it.
class PPODiscrete:  # (BaseAgent)
    """
    Minimal PPO for discrete action spaces (no GAE).
    - Separate actor/critic LRs
    - Rollout-based updates
    - Advantage normalization
    - Clipped objective
    - Optional entropy regularization
    """

    def __init__(self, state_dim, num_actions,
                 actor_lr=3e-4, critic_lr=3e-4, discount=0.99,
                 clip_epsilon=0.2, network_architecture=[64, 64],
                 update_epochs=10, mini_batch_size=64,
                 rollout_length=1024,
                 value_coef=0.5, entropy_coeff=0.01,
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

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build actor and critic
        self.policy_net = self._build_mlp(self.state_dim, self.num_actions, self.network_architecture).to(self.device)
        self.value_net  = self._build_mlp(self.state_dim, 1,               self.network_architecture).to(self.device)

        # Separate optimizers
        self.actor_opt  = optim.Adam(self.policy_net.parameters(), lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.value_net.parameters(),  lr=self.critic_lr)

        # Trajectory buffer: list of tuples
        # (state, action, reward, old_log_prob, value, done)
        self.trajectory = []

        # Cache for previous transition
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

    def start(self, new_state):
        # state: (state_dim,)
        s = self._to_tensor(new_state).unsqueeze(0)  # (1, state_dim)

        logits = self.policy_net(s)                  # (1, num_actions)
        dist = Categorical(logits=logits)
        action = dist.sample()                       # scalar tensor
        log_prob = dist.log_prob(action)             # scalar tensor

        value = self.value_net(s).squeeze(0).squeeze(-1)  # scalar tensor

        # Cache previous transition info
        self.prev_state = s.squeeze(0).detach().cpu().numpy()  # (state_dim,)
        self.prev_action = int(action.item())
        self.prev_log_prob = float(log_prob.item())
        self.prev_value = float(value.item())

        return self.prev_action

    def step(self, reward, new_state, done=False):
        # Store transition for the previous action
        self.trajectory.append((
            self.prev_state,
            int(self.prev_action),
            float(reward),
            float(self.prev_log_prob),
            float(self.prev_value),
            bool(done)
        ))

        # Next state forward
        s = self._to_tensor(new_state).unsqueeze(0)  # (1, state_dim)
        logits = self.policy_net(s)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        value = self.value_net(s).squeeze(0).squeeze(-1)

        # Update previous
        self.prev_state = s.squeeze(0).detach().cpu().numpy()
        self.prev_action = int(action.item())
        self.prev_log_prob = float(log_prob.item())
        self.prev_value = float(value.item())

        # Trigger PPO update if rollout length reached
        if len(self.trajectory) >= self.rollout_length:
            self._ppo_update()
            self.trajectory = []

        return self.prev_action

    def end(self, reward):
        # Terminal transition for the last action
        self.trajectory.append((
            self.prev_state,
            int(self.prev_action),
            float(reward),
            float(self.prev_log_prob),
            float(self.prev_value),
            True
        ))

        # Only update if rollout length reached
        if len(self.trajectory) >= self.rollout_length:
            self._ppo_update()
            self.trajectory = []

    def _ppo_update(self):
        traj = self.trajectory
        # Pack to tensors
        states = self._to_tensor(np.stack([t[0] for t in traj]))          # (T, state_dim)
        actions = torch.as_tensor([t[1] for t in traj], dtype=torch.int64, device=self.device)  # (T,)
        rewards = self._to_tensor([t[2] for t in traj])                    # (T,)
        old_log_probs = self._to_tensor([t[3] for t in traj])              # (T,)
        values = self._to_tensor([t[4] for t in traj])                     # (T,)
        dones = torch.as_tensor([t[5] for t in traj], dtype=torch.bool, device=self.device)     # (T,)

        # Compute discounted returns with bootstrap from last value if not terminal
        returns = self._compute_returns(rewards, dones, values)            # (T,)

        # Advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = states.shape[0]
        idx = torch.arange(T, device=self.device)

        for _ in range(self.update_epochs):
            perm = idx[torch.randperm(T)]
            for start in range(0, T, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = perm[start:end]
                if batch_idx.numel() == 0:
                    continue

                batch_states = states[batch_idx]           # (B, state_dim)
                batch_actions = actions[batch_idx]         # (B,)
                batch_old_log_probs = old_log_probs[batch_idx]  # (B,)
                batch_returns = returns[batch_idx]         # (B,)
                batch_advantages = advantages[batch_idx]   # (B,)

                # Actor forward
                logits = self.policy_net(batch_states)      # (B, num_actions)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)  # (B,)
                entropy = dist.entropy().mean()               # scalar

                # Ratios and clipped objective
                ratios = torch.exp(new_log_probs - batch_old_log_probs)  # (B,)
                obj1 = ratios * batch_advantages
                obj2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -(torch.min(obj1, obj2).mean() + self.entropy_coeff * entropy)

                # Critic loss (0.5 * MSE scaled by value_coef)
                values_pred = self.value_net(batch_states).squeeze(-1)    # (B,)
                value_err = values_pred - batch_returns
                critic_loss = self.value_coef * 0.5 * (value_err.pow(2).mean())

                # Optimize actor
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
                self.actor_opt.step()

                # Optimize critic
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
                self.critic_opt.step()

    def _compute_returns(self, rewards, dones, values):
        """
        Simple discounted returns with bootstrap from the last value if not terminal.
        Inputs: rewards (T,), dones (T,), values (T,)
        Output: returns (T,)
        """
        T = rewards.shape[0]
        returns = torch.zeros(T, dtype=torch.float32, device=self.device)
        G = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # Bootstrap from last value if trajectory didn't end in terminal
        if T > 0 and (not bool(dones[-1].item())):
            G = values[-1]

        for t in reversed(range(T)):
            if bool(dones[t].item()):
                G = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            G = rewards[t] + self.discount * G
            returns[t] = G
        return returns

    def reset(self):
        # Reinitialize networks and optimizers
        self.policy_net = self._build_mlp(self.state_dim, self.num_actions, self.network_architecture).to(self.device)
        self.value_net  = self._build_mlp(self.state_dim, 1,               self.network_architecture).to(self.device)
        self.actor_opt  = optim.Adam(self.policy_net.parameters(), lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.value_net.parameters(),  lr=self.critic_lr)
        self.trajectory = []
        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None