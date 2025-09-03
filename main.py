from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from solver.agent import DQNAgent
import os
import numpy as np
def main():
    yaml_path = "helper/config.yaml"
    loader = SUKPLoader(yaml_path)
    data = loader.get_data()
    param = loader.get_param()
    suk = SetUnionHandler(data, param)
    agent = DQNAgent(suk)
    
    episodes = 500
    batch_size = 64
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
        file_name = loader.get_filename()
        print("")
        print(f"Best result: {best_result}")
        print(f"Save result on result/{file_name}.npy")
        np.save(f"result/{file_name}.npy", best_sol)
        return
    finally:
        file_name = loader.get_filename()
        print(f"Best result: {best_result}")
        print(f"Save result on result/{file_name}.npy")
        np.save(f"result/{file_name}.npy", best_sol)

if __name__ == "__main__":
    main()