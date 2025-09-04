from collections import deque
from networks.dqn import DeepQlearningNetwork
from helper.set_handler import SetUnionHandler
import numpy as np
import torch
if torch.cuda.is_available():
    torch.set_default_device('cuda')
import torch.nn as nn
import torch.optim as optim


class DQNAgent:
    def __init__(self, env: SetUnionHandler, device, load_checkpoint, file_name):
        self.state_size = env.m
        self.action_size = env.m + 1   #an action represent for terminate
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99997
        self.learning_rate = 0.001
        self.rng = np.random.default_rng(42)
        self.env = env
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
        if self.rng.random() <= self.epsilon:
            return self.rng.integers(self.action_size)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            action_values = self.model(state_tensor)
            return action_values.argmax().item()
        return None
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0
        indies = self.rng.choice(len(self.memory), size=batch_size, replace=False)
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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return average_loss
    
    
        

