"""

"""

import numpy as np
from copy import deepcopy

from rlforge.policies import softmax
from rlforge.agents import BaseAgent
from rlforge.function_approximation import MLP
from rlforge.utils import ExperienceBuffer
from rlforge.optimizers import AdamOptimizer

class DQNAgent(BaseAgent):

    def __init__(self, learning_rate, discount, state_dim, num_actions, temperature=1, 
                 network_architecture=[2], target_network_update_steps=8,
                 num_replay=0, experience_buffer_size=1024, mini_batch_size=8):
        
        self.learning_rate = learning_rate
        self.discount = discount
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.temperature = temperature

        self.network_architecture = network_architecture
        self.main_network = MLP(input_dim=self.state_dim,
                                output_dim=self.num_actions,
                                hidden_layers=self.network_architecture,
                                learning_rate=self.learning_rate)
        self.target_network = deepcopy(self.main_network)
        self.target_network_update_steps = target_network_update_steps
        self.elapsed_training_steps = 0

        self.num_replay = num_replay
        self.experience_buffer_size = experience_buffer_size
        self.mini_batch_size = mini_batch_size
        self.experience_buffer = ExperienceBuffer(self.experience_buffer_size, self.mini_batch_size)

    def start(self, new_state):

        q_values, _ = self.main_network.forward_propagation(new_state)
        action = self.select_action(q_values, self.temperature)

        self.prev_state = new_state
        self.prev_action = action

        return action
    
    def step(self, reward, new_state):

        q_values, _ = self.main_network.forward_propagation(new_state)
        action = self.select_action(q_values, self.temperature)
        
        self.experience_buffer.append(self.prev_state, self.prev_action, reward, 0, new_state)

        if len(self.experience_buffer.buffer) > self.experience_buffer.mini_batch_size:
            for _ in range(self.num_replay):
            
                experiences = self.experience_buffer.sample()

                td_errors, cache = self.get_td_error(experiences)

                grads = self.main_network.backward_propagation(self.mini_batch_size, td_errors, cache)
                self.main_network.update_weights(grads)

        self.elapsed_training_steps += 1

        if self.elapsed_training_steps == self.target_network_update_steps:
            self.target_network = deepcopy(self.main_network)
            self.elapsed_training_steps = 0

        self.prev_state = new_state
        self.prev_action = action

        return action
    
    def end(self, reward):
    
        new_state = np.zeros_like(self.prev_state)

        self.experience_buffer.append(self.prev_state, self.prev_action, reward, 1, new_state)
        
        if len(self.experience_buffer.buffer) > self.experience_buffer.mini_batch_size:
            for _ in range(self.num_replay):
            
                experiences = self.experience_buffer.sample()

                td_errors, cache = self.get_td_error(experiences)

                grads = self.main_network.backward_propagation(self.mini_batch_size, td_errors, cache)
                self.main_network.update_weights(grads)

    def select_action(self, q_values, temperature):
        
        softmax_probs = softmax(q_values, temperature)
        action = np.random.choice(self.num_actions, p=softmax_probs)

        return action
    
    def get_td_error(self, experiences):

        states, actions, rewards, terminal, new_states = map(list, zip(*experiences))
        states = np.vstack(states)
        actions = np.vstack(actions).squeeze()
        rewards = np.vstack(rewards).squeeze()
        terminal = np.vstack(terminal).squeeze()
        new_states = np.vstack(new_states)

        target_q_values, _ = self.main_network.forward_propagation(new_states)
        q_values, cache = self.target_network.forward_propagation(states)
        
        indices = np.arange(self.mini_batch_size)
        q_values_vec = q_values[indices,actions]

        td_error = rewards + self.discount*np.max(target_q_values,axis=1)*(1 - terminal) - q_values_vec
        
        td_error_mat = np.zeros((self.mini_batch_size,self.num_actions))
        td_error_mat[indices,actions] = td_error
        
        return td_error_mat, cache
    
    def reset(self):

        self.main_network = MLP(input_dim=self.state_dim,
                                output_dim=self.num_actions,
                                hidden_layers=self.network_architecture,
                                learning_rate=self.learning_rate)
        self.target_network = deepcopy(self.main_network)
        self.experience_buffer = ExperienceBuffer(self.experience_buffer_size, self.mini_batch_size)
        self.elapsed_training_steps = 0