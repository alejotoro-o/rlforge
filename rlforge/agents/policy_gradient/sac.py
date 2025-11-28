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
    Soft Actor-Critic (SAC) Agent for continuous action spaces.

    SAC is an off-policy actor-critic algorithm that optimizes a stochastic
    policy in an entropy-regularized reinforcement learning framework. It
    balances exploration and exploitation by maximizing both expected reward
    and policy entropy.

    This implementation builds all networks internally for proper reset and
    management, including policy, twin Q-networks, and entropy tuning.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state space.
    action_dim : int
        Dimension of the continuous action space.
    policy_net_architecture : tuple of int, optional
        Hidden layer sizes for the policy network (default=(64, 64)).
    q_net_architecture : tuple of int, optional
        Hidden layer sizes for the Q-networks (default=(64, 64)).
    actor_lr : float, optional
        Learning rate for the actor/policy network (default=3e-4).
    critic_lr : float, optional
        Learning rate for the critic/Q-networks (default=3e-4).
    alpha_lr : float, optional
        Learning rate for the entropy coefficient α (default=3e-4).
    discount : float, optional
        Discount factor γ applied to future rewards (default=0.99).
    tau : float, optional
        Polyak averaging factor for soft target network updates (default=0.005).
    update_frequency : int, optional
        Frequency (in steps) of training updates (default=1).
    buffer_size : int, optional
        Maximum size of the replay buffer (default=1,000,000).
    mini_batch_size : int, optional
        Size of mini-batches sampled from the replay buffer (default=256).
    update_start_size : int, optional
        Minimum number of transitions before updates begin (default=256).
    tanh_squash : bool, optional
        Whether to apply tanh squashing to actions (default=True).
    action_low : float or np.ndarray, optional
        Lower bound(s) for continuous actions.
    action_high : float or np.ndarray, optional
        Upper bound(s) for continuous actions.
    target_entropy_factor : float, optional
        Factor for target entropy calculation (default=0.9).
    device : str or torch.device, optional
        Device to run computations on ("cpu" or "cuda"). Defaults to CUDA if available.
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
        """
        Initialize weights for linear layers.

        Uses Kaiming uniform initialization for weights and sets biases to zero.
        Suitable for Tanh/ReLU activations.
        """

        if isinstance(m, nn.Linear):
            # Use Kaiming He initialization (standard for Tanh/ReLU)
            nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _create_network(self, input_dim, output_dim, architecture):
        """
        Build a standard feedforward MLP network.

        Constructs a sequential model with Tanh activations in hidden layers.

        Parameters
        ----------
        input_dim : int
            Dimension of the input features.
        output_dim : int
            Dimension of the output (e.g., action_dim or 1 for Q-value).
        architecture : tuple of int
            Sizes of hidden layers.

        Returns
        -------
        nn.Sequential
            The constructed PyTorch network.
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
        """
        Move network to device and set training/evaluation mode.

        Parameters
        ----------
        net : nn.Module
            The network to configure.
        requires_grad : bool
            If True, enables training mode and gradients; otherwise sets eval mode
            and disables gradients.
        """

        net.to(self.device)
        net.train() if requires_grad else net.eval()
        for param in net.parameters():
            param.requires_grad = requires_grad

    def _to_tensor(self, x):
        """
        Convert input to a float32 tensor on the agent's device.

        Parameters
        ----------
        x : array-like or scalar
            Input data.

        Returns
        -------
        torch.Tensor
            Float tensor on the agent's device.
        """

        return torch.as_tensor(x, dtype=torch.float32, device=self.device)
    
    def _q_net_forward(self, q_net, state, action):
        """
        Forward pass through a Q-network.

        Concatenates state and action before passing through the Q-network.

        Parameters
        ----------
        q_net : nn.Module
            The Q-network to evaluate.
        state : torch.Tensor
            Batch of states.
        action : torch.Tensor
            Batch of actions.

        Returns
        -------
        torch.Tensor
            Predicted Q-values.
        """

        sa = torch.cat([state, action], dim=-1)
        return q_net(sa)

    def _sample_action(self, mean, deterministic=False):
        """
        Sample an action from the policy distribution.

        Uses a Normal distribution with learned mean and log standard deviation.
        Supports reparameterization for stochastic sampling and optional tanh
        squashing with log-prob correction.

        Parameters
        ----------
        mean : torch.Tensor
            Policy network output (mean action).
        deterministic : bool, optional
            If True, returns mean action without noise (default=False).

        Returns
        -------
        tuple
            (action, log_prob, raw_z) where:
            - action : torch.Tensor, final action after squashing/rescaling
            - log_prob : torch.Tensor, log probability of the sampled action
            - raw_z : torch.Tensor, unsquashed latent sample
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

    def start(self, state, deterministic=False):
        """
        Begin a new episode in a single environment.

        Parameters
        ----------
        state : array-like
            Initial state of the environment.
        deterministic : bool, optional
            If True, selects deterministic actions (default=False).

        Returns
        -------
        np.ndarray
            Selected action.
        """

        actions = self.start_batch(np.expand_dims(state, axis=0), deterministic)
        return actions[0]

    def step(self, reward, state, done=False, deterministic=False):
        """
        Take a step in a single environment.

        Stores transition, performs updates if conditions are met,
        and selects the next action.

        Parameters
        ----------
        reward : float
            Reward from the previous action.
        state : array-like
            Next state observed.
        done : bool, optional
            Whether the episode has terminated (default=False).
        deterministic : bool, optional
            If True, selects deterministic actions (default=False).

        Returns
        -------
        np.ndarray
            Selected action.
        """

        actions = self.step_batch(
            np.array([reward], dtype=np.float32),
            np.expand_dims(state, axis=0),
            np.array([done], dtype=np.bool_),
            deterministic
        )
        return actions[0]

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


    def start_batch(self, states, deterministic=False):
        """
        Begin a batch of episodes.

        Selects actions for multiple environments simultaneously.

        Parameters
        ----------
        states : array-like, shape (N, state_dim)
            Batch of initial states.
        deterministic : bool, optional
            If True, selects deterministic actions (default=False).

        Returns
        -------
        np.ndarray
            Array of selected actions of shape (N, action_dim).
        """

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
        """
        Take a step in multiple environments.

        Stores transitions in the replay buffer, performs SAC updates if
        conditions are met, and selects next actions.

        Parameters
        ----------
        rewards : array-like, shape (N,)
            Rewards from the previous actions.
        next_states : array-like, shape (N, state_dim)
            Next states observed.
        dones : array-like, shape (N,)
            Boolean flags indicating episode termination.
        deterministic : bool, optional
            If True, selects deterministic actions (default=False).

        Returns
        -------
        np.ndarray
            Array of selected actions of shape (N, action_dim).
        """

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
        """
        Complete a batch of episodes.

        Stores terminal transitions into the replay buffer and performs SAC
        updates if conditions are met.

        Parameters
        ----------
        rewards : array-like, shape (N,)
            Final rewards received for each terminated environment.
        """

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
        """
        Core SAC update logic.

        Performs the main training loop for Soft Actor-Critic, including critic
        updates, actor updates, entropy coefficient (α) updates, and Polyak
        averaging for target networks.

        Workflow
        --------
        1. Sample a mini-batch of transitions from the replay buffer.
        2. Critic update:
        - Compute target Q-values using target critics and target policy.
        - Apply entropy regularization: subtract α * logπ(a'|s').
        - Minimize MSE loss for both Q-networks.
        3. Actor update:
        - Sample actions from the current policy.
        - Minimize expected loss: E[-min(Q(s,a)) + α * logπ(a|s)].
        4. Temperature/α update:
        - Adjust α to match target entropy.
        - Optimize logα parameter.
        5. Target network soft update:
        - Polyak averaging for both target Q-networks.

        Notes
        -----
        - Critic updates occur at every training step.
        - Actor and α updates are performed alongside critic updates.
        - Target networks are updated using Polyak averaging.

        Returns
        -------
        None
            Updates actor, critics, and entropy coefficient in-place.
        """


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
        Build or rebuild all networks and optimizers.

        Initializes the policy network, twin Q-networks, target Q-networks,
        and learnable parameters for log standard deviation and log α.
        Also sets up optimizers for actor, critics, and α.

        Parameters
        ----------
        target_entropy_factor : float, optional
            Factor used to compute target entropy (default=0.9).
        init_weights : bool, optional
            If True, initializes target entropy based on action_dim (default=False).

        Workflow
        --------
        1. Construct policy network (outputs mean actions).
        2. Construct twin Q-networks (Q1 and Q2).
        3. Deep copy Q-networks to create target critics.
        4. Initialize learnable parameters: log_std and log_alpha.
        5. Compute target entropy if init_weights=True.
        6. Initialize Adam optimizers for actor, critics, and α.
        7. Update α from logα.

        Returns
        -------
        None
            Networks, parameters, and optimizers are rebuilt in-place.
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
        Reset the agent state for a new run.

        Clears the replay buffer, resets counters, and rebuilds networks
        and optimizers to start training from scratch.

        Notes
        -----
        - Resets ``total_steps`` to zero.
        - Clears cached previous state and action.
        - Calls ``reset_nets_and_opts()`` to reinitialize networks and optimizers.

        Returns
        -------
        None
            Agent state and networks are reset.
        """

        # Rebuild all nets, re-init optimizers, log_std, and log_alpha
        self.reset_nets_and_opts()
        
        # Reset buffers and tracking state
        self.replay_buffer.clear()
        self.total_steps = 0
        self.prev_state = None
        self.prev_action = None