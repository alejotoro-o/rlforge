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
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent for continuous action spaces.

    TD3 enhances the Deep Deterministic Policy Gradient (DDPG) algorithm with three
    core mechanisms:
    - Twin Critics: Two Q-networks to reduce overestimation bias.
    - Delayed Policy Updates: The actor (policy) is updated less frequently than the critics.
    - Target Policy Smoothing: Adds clipped noise to target actions for more stable training.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state space.
    action_dim : int
        Dimension of the continuous action space.
    policy_net_architecture : tuple of int, optional
        Hidden layer sizes for the actor/policy network (default=(256, 256)).
    q_net_architecture : tuple of int, optional
        Hidden layer sizes for the critic/Q-networks (default=(256, 256)).
    actor_lr : float, optional
        Learning rate for the actor network (default=1e-4).
    critic_lr : float, optional
        Learning rate for the critic networks (default=1e-3).
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
    action_low : float or np.ndarray, optional
        Lower bound(s) for continuous actions.
    action_high : float or np.ndarray, optional
        Upper bound(s) for continuous actions.
    noise_std : float, optional
        Standard deviation of Gaussian exploration noise added to actions (default=0.1).
    policy_delay : int, optional
        Delay factor for policy and target network updates (default=2).
    target_noise_std : float, optional
        Standard deviation of noise added to target actions during critic updates (default=0.2).
    target_noise_clip : float, optional
        Clipping value for target action noise (default=0.5).
    device : str or torch.device, optional
        Device to run computations on ("cpu" or "cuda"). Defaults to CUDA if available.
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
        """
        Initialize weights for linear layers.

        Uses Kaiming uniform initialization for weights and sets biases to zero
        for stability in ReLU/Tanh networks.

        Parameters
        ----------
        m : nn.Module
            PyTorch module (typically nn.Linear) to initialize.
        """

        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _create_network(self, input_dim, output_dim, architecture, final_activation=None):
        """
        Build a standard feedforward MLP network.

        Constructs a sequential model with ReLU activations in hidden layers
        and an optional final activation.

        Parameters
        ----------
        input_dim : int
            Dimension of the input features.
        output_dim : int
            Dimension of the output (e.g., action_dim or 1 for Q-value).
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

    def _sample_action(self, mean, deterministic=False, action_low_np=None, action_high_np=None):
        """
        Sample and scale an action from the policy output.

        Rescales the policy output from [-1, 1] to environment bounds,
        optionally adds Gaussian exploration noise, and clips to bounds.

        Parameters
        ----------
        mean : torch.Tensor
            Policy network output (mean action in [-1, 1]).
        deterministic : bool, optional
            If True, returns mean action without noise (default=False).
        action_low_np : np.ndarray
            Lower bounds for actions.
        action_high_np : np.ndarray
            Upper bounds for actions.

        Returns
        -------
        torch.Tensor
            Final action tensor clipped to environment bounds.
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

    # --- Batch implementation ---
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
        """
        Take a step in multiple environments.

        Stores transitions in the replay buffer, performs TD3 updates if
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
        """
        Complete a batch of episodes.

        Stores terminal transitions into the replay buffer and performs TD3
        updates if conditions are met.

        Parameters
        ----------
        rewards : array-like, shape (N,)
            Final rewards received for each terminated environment.
        """

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
        """
        Core TD3 update logic.

        Performs the main training loop for TD3, including critic updates,
        delayed actor updates, and Polyak averaging for target networks.

        Workflow
        --------
        1. Sample a mini-batch of transitions from the replay buffer.
        2. Compute target Q-values using target policy and target critics:
        - Apply target policy smoothing by adding clipped Gaussian noise
            to target actions.
        - Use clipped double Q-learning (min(Q1, Q2)) to reduce bias.
        3. Update both critics (Q1 and Q2) by minimizing MSE loss against
        the target Q-values.
        4. Every ``policy_delay`` steps:
        - Update the actor by maximizing Q1(s, π(s)).
        - Soft-update target networks using Polyak averaging.

        Notes
        -----
        - Critic updates occur at every training step.
        - Actor and target network updates are delayed by ``policy_delay``.
        - Target actions are clipped to the environment's action bounds.

        Returns
        -------
        None
            Updates actor and critic networks in-place.
        """

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
        Build or rebuild all networks and optimizers.

        Initializes the policy network, twin Q-networks, and their target
        counterparts. Also sets up optimizers for the actor and both critics.

        Workflow
        --------
        1. Construct the policy network with Tanh activation on the output.
        2. Construct twin Q-networks (Q1 and Q2).
        3. Deep copy networks to create target policy and target critics.
        4. Set target networks to evaluation mode (no gradient updates).
        5. Initialize Adam optimizers for actor and critics.

        Returns
        -------
        None
            Networks and optimizers are rebuilt in-place.
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
        Reset the agent state for a new run.

        Clears the replay buffer, resets counters, and rebuilds networks
        and optimizers to start training from scratch.

        Notes
        -----
        - Resets ``total_steps`` to zero.
        - Clears cached previous state and action.
        - Calls ``reset_nets_and_opts()`` to reinitialize networks.

        Returns
        -------
        None
            Agent state and networks are reset.
        """

        self.reset_nets_and_opts()

        self.replay_buffer.clear()
        self.total_steps = 0
        self.prev_state = None
        self.prev_action = None