from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from helper.k_heapq import TopKHeap
from helper.generate_initial import random_gen
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
    ran_sol_l, ran_prof_l = [], []
    for _ in range(100):
        ran_sol, ran_prof = random_gen(suk)
        ran_sol_l.append(ran_sol)
        ran_prof_l.append(ran_prof)
    print(ran_prof_l)
    suk.set_init_sol(ran_sol_l)
    best_result = 0
    best_sol = np.array([])
    try:
        for e in range(episodes):
            print(f"Start episode {e + 1}")
            suk.reset_init()
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
            print(f"Episode {e+1}, Reward: {total_reward}, Result: {best_result}, Loss: {loss}, total step: {count}")
    except KeyboardInterrupt:
        print("")
        print(f"Best result: {best_result}")
        print(f"Save result on result/{file_name}.npy")
        np.save(f"result/{file_name}.npy", best_sol)
        if save_checkpoint_:
            save_checkpoint(best_result, file_name, best_sol, agent)
        return
    finally:
        solution_list = heap.get_top_k_states()
        solution_profit = heap.get_top_k_values()
        print(solution_profit)
        for i in range(20):
            current_solution_list = solution_list.copy()  # Run 10 times for diversity
            current_profit_list = solution_profit.copy()
            for _ in range(20): 
                current_best_sol, current_best_prof = suk.iterated_local_search(solution_list, current_solution_list, current_profit_list)
            print(f"Loop {i}: Current profit: {current_profit_list}, \n Best profit: {current_best_prof}")
            best_sol = current_best_sol
            solution_list = current_solution_list
            solution_profit = current_profit_list
        max_profit = max(solution_profit)
        best_sol = solution_list[solution_profit.index(max_profit)]
        result_str = ' '.join(['1' if x > 0.5 else '0' for x in best_sol])
        print(f"Result: {result_str}")
        suk.set_state(best_sol)
        print(f"Total weight: {suk.get_weight()}, capacity: {loader.capacity}")
        print(f"After ILS: Max Profit: {max_profit}, Best sol: {best_sol}")
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
