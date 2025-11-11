import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from rlforge.policies import softmax
from rlforge.agents import BaseAgent
from rlforge.utils import ExperienceBuffer


class DQNTorchAgent(BaseAgent):
    """
    Deep Q-Network (DQN) Agent implemented with PyTorch.

    This agent allows passing any PyTorch model (MLP, CNN, Transformer, etc.)
    as the function approximator. It uses an experience replay buffer and a
    target network for stabilizing training.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model mapping states -> Q-values.
    learning_rate : float
        Learning rate for the optimizer.
    discount : float
        Discount factor (gamma).
    num_actions : int
        Number of discrete actions.
    temperature : float, optional (default=1)
        Softmax temperature for exploration.
    target_network_update_steps : int, optional (default=8)
        Frequency (in training steps) to copy weights to target network.
    num_replay : int, optional (default=0)
        Number of replay updates per environment step.
    experience_buffer_size : int, optional (default=1024)
        Maximum size of the replay buffer.
    mini_batch_size : int, optional (default=8)
        Number of samples per replay update.
    device : str, optional (default="cpu")
        Device to run the model on ("cpu" or "cuda").
    """

    def __init__(self, model, learning_rate, discount, num_actions,
                 temperature=1, target_network_update_steps=8,
                 num_replay=0, experience_buffer_size=1024, mini_batch_size=8,
                 device="cpu"):

        self.discount = discount
        self.num_actions = num_actions
        self.temperature = temperature
        self.target_network_update_steps = target_network_update_steps
        self.num_replay = num_replay
        self.mini_batch_size = mini_batch_size
        self.device = device

        # Main and target networks
        self.main_network = model.to(self.device)
        self.target_network = deepcopy(model).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.experience_buffer = ExperienceBuffer(experience_buffer_size, mini_batch_size)

        self.elapsed_training_steps = 0

    def start(self, new_state):
        state_tensor = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.main_network(state_tensor).detach().cpu().numpy()
        action = self.select_action(q_values, self.temperature)

        self.prev_state = new_state
        self.prev_action = action
        return action

    def step(self, reward, new_state):
        state_tensor = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.main_network(state_tensor).detach().cpu().numpy()
        action = self.select_action(q_values, self.temperature)

        self.experience_buffer.append(self.prev_state, self.prev_action, reward, 0, new_state)

        if len(self.experience_buffer.buffer) > self.mini_batch_size:
            for _ in range(self.num_replay):
                self._train_step()

        self.elapsed_training_steps += 1
        if self.elapsed_training_steps == self.target_network_update_steps:
            self.target_network.load_state_dict(self.main_network.state_dict())
            self.elapsed_training_steps = 0

        self.prev_state = new_state
        self.prev_action = action
        return action

    def end(self, reward):
        new_state = np.zeros_like(self.prev_state)
        self.experience_buffer.append(self.prev_state, self.prev_action, reward, 1, new_state)

        if len(self.experience_buffer.buffer) > self.mini_batch_size:
            for _ in range(self.num_replay):
                self._train_step()

    def select_action(self, q_values, temperature):
        softmax_probs = softmax(q_values, temperature)
        action = np.random.choice(self.num_actions, p=softmax_probs)
        return action

    def _train_step(self):
        states, actions, rewards, terminal, new_states = map(list, zip(*self.experience_buffer.sample()))

        states = torch.tensor(np.vstack(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.vstack(actions).squeeze(), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.vstack(rewards).squeeze(), dtype=torch.float32).to(self.device)
        terminal = torch.tensor(np.vstack(terminal).squeeze(), dtype=torch.float32).to(self.device)
        new_states = torch.tensor(np.vstack(new_states), dtype=torch.float32).to(self.device)

        # Q(s,a)
        q_values = self.main_network(states)
        q_values_vec = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q(s',a')
        with torch.no_grad():
            target_q_values = self.target_network(new_states)
            max_next_q = target_q_values.max(1)[0]
            target = rewards + self.discount * max_next_q * (1 - terminal)

        loss = nn.MSELoss()(q_values_vec, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset(self):
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.experience_buffer = ExperienceBuffer(self.experience_buffer.size, self.mini_batch_size)
        self.elapsed_training_steps = 0