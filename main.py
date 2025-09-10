from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from helper.k_heapq import TopKHeap
from solver.softmax_agent import DQNAgent
import torch
import numpy as np



def main():
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
    episodes = config['episodes']
    batch_size = config['batch_size']
    file_name = loader.get_filename()
    suk = SetUnionHandler(data, param)
    agent = DQNAgent(suk, device, load_checkpoint_, file_name)
    heap = TopKHeap(100)
    best_result = 0
    best_sol = np.array([])
    try:
        for e in range(episodes):
            print(f"Start episode {e + 1}")
            suk.reset()
            # print(f"Init solution: Profit {suk.get_profit()}, Weight {suk.get_weight()}")
            state = suk.get_state()  # Assume env has reset/step
            terminate = False
            total_reward = 0.0
            loss = 0.0
            count = 0
            while not terminate:
                count += 1
                action = agent.action(state)
                next_state, reward, terminate = suk.step(action)  # Adjust to your env's signature
                if np.isnan(reward):
                    raise ValueError(f"nan reward {reward}")
                agent.remember(state, action, reward, next_state, terminate)
                heap.add(suk.get_profit(), suk.get_state())
                state = next_state
                total_reward += reward
                if suk.get_profit() > best_result:
                    best_result = suk.get_profit()
                    best_sol = suk.get_state()
                loss += agent.replay(batch_size)
            loss = loss / count
            print(f"Episode {e+1}, Reward: {total_reward}, Result: {best_result}, Loss: {loss}, terminate prob: {agent.terminate_probality}, total step: {count}")
    except KeyboardInterrupt:
        print("")
        print(f"Best result: {best_result}")
        print(f"Save result on result/{file_name}.npy")
        np.save(f"result/{file_name}.npy", best_sol)
        if save_checkpoint_:
            save_checkpoint(best_result, file_name, best_sol, agent)
        return
    finally:
        k_results = heap.get_top_k_states()
        for i in range(200):
            if i % 10 == 0:
                print("ILS phase {i}: Max Profit: {best_result}, Best sol: {best_sol}")
            tem_best_sol, tem_best_result = suk.iterated_local_search(k_results)
            if tem_best_result > best_result:
                best_result = tem_best_result
                best_sol = tem_best_sol
        print(f"After ILS: Max Profit: {best_result}, Best sol: {best_sol}")
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
