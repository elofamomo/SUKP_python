from collections import deque
from networks.dqn100 import DeepQlearningNetwork
from helper.set_handler import SetUnionHandler
import numpy as np
import torch
if torch.cuda.is_available():
    torch.set_default_device('cuda')
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist


class DQNAgent:
    def __init__(self, env: SetUnionHandler, device, load_checkpoint, file_name):
        self.state_size = env.m
        self.action_size = 2 * env.m + 1   #an action represent for terminate
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.terminate_probality = 0.0
        self.learning_rate = 0.001
        self.rng = np.random.default_rng(42)
        self.env = env
        self.tabu_size = self.env.tabu_size
        self.tabu = {}
        self.noise_std = self.env.noise_std
        self.noise_decay = self.env.noise_decay
        self.device = device
        self.load_checkpoint = load_checkpoint
        self.model = DeepQlearningNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = DeepQlearningNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.load_checkpoint:
            checkpoint = torch.load(f'checkpoints/{file_name}.pth', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, terminate):
        self.memory.append((state, action, reward, next_state, terminate))

    def action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_values = self.model(state_tensor)
        self.set_valid_action(action_values)
        action_values[self.action_size - 1] = float('-inf')
        noise = torch.rand_like(action_values) * self.noise_std
        action_values = action_values + noise
        log_probs = torch.log_softmax(action_values / self.env.tau, dim=0)
        softmax_torch = torch.exp(log_probs)
        self.terminate_probability = softmax_torch[self.action_size - 1].item()
        action_dist = dist.Categorical(logits=log_probs)
        action = action_dist.sample().item()
        if 0 <= action and action < self.state_size:
            print(f"Add: {action}")
        elif self.state_size <= action and action < 2 * self.state_size:
            print(f"remove: {action - self.state_size}")
        entropy = -torch.sum(softmax_torch * torch.log(softmax_torch + 1e-8)).item()
        return action, entropy
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0
        current_batch_size = batch_size
        indies = self.rng.choice(len(self.memory), size=current_batch_size, replace=False)
        minibatch = [self.memory[i] for i in indies]
        total_loss = 0.0
        for state, action, reward, next_state, terminate in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            target = reward 
            if not terminate:
                target += self.gamma * self.target_model(next_state_tensor).max().item()
            target_f = self.model(state_tensor)
            target_f = target_f.clone()
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()
            total_loss += loss
        average_loss = total_loss / batch_size
        return average_loss
    
    def calcualte_batch_size(self, epsilon, batch_size):
        res = 0 
        rate = (epsilon - self.epsilon_min) / (self.epsilon_max - self.epsilon_min)
        res = rate * (batch_size - 32) + 32
        return round(res)
    
    def softmax(self, x, axis=-1):
        # Subtract the max for numerical stability
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        # Compute softmax values
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    def update_tabu(self, action):
        if 0 <= action and action < self.state_size:
            self.tabu[action + self.state_size] = self.tabu_size
        elif self.state_size <= action and action < 2 * self.state_size:
            self.tabu[action - self.state_size] = self.tabu_size
    
    def decay(self):
        self.decay_tabu()
        self.decay_noise()

    def decay_noise(self):
        self.noise_std = max(0.01, self.noise_std * self.noise_decay)

    def decay_tabu(self):
        need_remove = []
        for item in self.tabu:
            self.tabu[item] -= 1
            if self.tabu[item] <= 0:
                need_remove.append(item)
        for item in need_remove:
            del self.tabu[item]
    
    def set_valid_action(self, action_values):
        for action in range(self.state_size):
            if action in self.env.selected_items or action in self.tabu:
                action_values[action] = float('-inf')
            else:
                marginal_weight = sum(self.env.element_weights[elem] for elem in self.env.item_subsets[action] if self.env.element_counts[elem] == 0)
                if self.env.total_weight + marginal_weight > self.env.capacity:
                    action_values[action] = float('-inf')

        for action in range(self.state_size, 2 * self.state_size):
            if action - self.state_size not in self.env.selected_items or action in self.tabu:
                action_values[action] = float('-inf')

