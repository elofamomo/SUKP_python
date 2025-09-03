from networks.dqn import DeepQlearningNetwork
import torch

net = DeepQlearningNetwork(input_size=4, output_size=2)
input = torch.randn(1, 4)
print(input)
output = net(input)
print(output)