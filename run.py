from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from helper.k_heapq import TopKHeap
from helper.generate_initial import ga_solver, hamming_distance
from solver.softmax_agent import DQNAgent
from metric.plotter import Plotter
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import torch.optim as optim
from networks.dqn200 import DeepQlearningNetwork


if torch.cuda.is_available():
        torch.set_default_device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using: {device}")
yaml_path = "helper/config.yaml"
loader = SUKPLoader(yaml_path)
data = loader.get_data()
param = loader.get_param()
config = loader.load_general_config()
print(config)
save_checkpoint_ = config['save_checkpoint']
load_checkpoint_ = config['load_checkpoint']
update_interval = config['update_interval']
episodes = config['episodes']
batch_size = config['batch_size']
file_name = loader.get_filename()
suk = SetUnionHandler(data, param)
network = DeepQlearningNetwork(suk.m, 2 * suk.m + 1).to(device) 
checkpoint = torch.load(f'checkpoints/{file_name}.pth', weights_only=False)
network.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.Adam(network.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

