from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from solver.agent import DQNAgent
import torch
import numpy as np


def main():
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yaml_path = "helper/config.yaml"
    loader = SUKPLoader(yaml_path)
    data = loader.get_data()
    param = loader.get_param()
    config = loader.load_general_config()
    print(config)
    save_checkpoint_ = config['save_checkpoint']
    load_checkpoint_ = config['load_checkpoint']
    episodes = config['episodes']
    batch_size = config['batch_size']
    file_name = loader.get_filename()
    suk = SetUnionHandler(data, param)
    agent = DQNAgent(suk, device, load_checkpoint_, file_name)
    

    best_result = 0
    best_sol = np.array([])
    try:
        for e in range(episodes):
            print(f"Start episode {e + 1}")
            suk.reset()
            state = suk.get_state()  # Assume env has reset/step
            terminate = False
            total_reward = 0.0
            while not terminate:
                action = agent.action(state)
                next_state, reward, terminate = suk.step(action)  # Adjust to your env's signature
                if np.isnan(reward):
                    raise ValueError(f"nan reward {reward}")
                agent.remember(state, action, reward, next_state, terminate)
                state = next_state
                total_reward += reward
                if total_reward > best_result:
                    best_result = suk.get_profit()
                    best_sol = suk.get_state()
                agent.replay(batch_size)
            print(f"Episode {e+1}, Reward: {total_reward}, Result: {best_result}, Epsilon: {agent.epsilon}")
    except KeyboardInterrupt:
        print("")
        print(f"Best result: {best_result}")
        print(f"Save result on result/{file_name}.npy")
        np.save(f"result/{file_name}.npy", best_sol)
        if save_checkpoint_:
            save_checkpoint(best_result, file_name, best_sol, agent)
        return
    finally:
        print(f"Best result: {best_result}")
        print(f"Save result on result/{file_name}.npy")
        np.save(f"result/{file_name}.npy", best_sol)
        if save_checkpoint_:
            save_checkpoint(best_result, file_name, best_sol, agent)

def save_checkpoint(best_result, file_name, best_sol, agent):
    checkpoint = {
        'best_result': best_result,
        'best_sol': best_sol,
        'model_state_dict': agent.model.state_dict(),
        'target_model_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict()
    }
    torch.save(checkpoint, f'checkpoints/{file_name}.pth')


if __name__ == "__main__":
    main()