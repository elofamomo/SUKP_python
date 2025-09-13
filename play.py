import torch
import numpy as np
from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from networks.dqn100 import DeepQlearningNetwork

def play(yaml_path="helper/config.yaml", checkpoint_path=None):
    """
    Replays the trained DQN policy from a checkpoint file and evaluates the results.
    
    :param yaml_path: str, path to the YAML config file for loading SUKP data
    :param checkpoint_path: str, path to the checkpoint file (.pth)
    :return: tuple (best_solution: np.array, best_profit: float, total_weight: float)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load SUKP data
    loader = SUKPLoader(yaml_path)
    data = loader.get_data()
    param = loader.get_param()
    file_name = loader.get_filename()
    if checkpoint_path is None:
        checkpoint_path = f'checkpoints/{file_name}.pth'

    # Initialize environment
    suk = SetUnionHandler(data, param)

    # Initialize DQN model
    state_size = suk.m
    action_size = 2 * suk.m + 1  # Matches your DQNAgent setup
    model = DeepQlearningNetwork(state_size, action_size).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode (disables dropout if any)

    # Replay policy
    suk.reset()
    state = suk.get_state()
    total_reward = 0.0
    terminate = False
    steps = 0
    best_solution = state.copy()
    best_profit = 0.0

    while steps < 10000:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_values = model(state_tensor)
            log_probs = torch.log_softmax(action_values, dim=0)
            action_values[2 * suk.m] = float('-inf') 
            action = torch.argmax(action_values).item()  # Greedy action (argmax for evaluation)
        
        next_state, reward, terminate = suk.step(action)
        total_reward += reward
        state = next_state
        print(steps)
        steps += 1

        current_profit = suk.get_profit()
        if current_profit > best_profit:
            best_profit = current_profit
            best_solution = suk.get_state().copy()

        

    total_weight = suk.get_weight()
    print(f"Replay Results:")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Best Profit: {best_profit:.2f}")
    print(f"Total Weight: {total_weight:.2f} (Capacity: {suk.capacity})")
    print(f"Feasible: {suk.is_feasible()}")
    print(f"Solution: {' '.join(['1' if x > 0.5 else '0' for x in best_solution])}")

    return best_solution, best_profit, total_weight

if __name__ == "__main__":
    play()