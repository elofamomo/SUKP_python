from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from helper.k_heapq import TopKHeap
from helper.generate_initial import ga_solver, hamming_distance
from solver.softmax_agent import DQNAgent
from metric.plotter import Plotter
from torch.utils.tensorboard import SummaryWriter
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
    update_interval = config['update_interval']
    episodes = config['episodes']
    batch_size = config['batch_size']
    file_name = loader.get_filename()
    suk = SetUnionHandler(data, param)
    agent = DQNAgent(suk, device, load_checkpoint_, file_name)
    heap = TopKHeap(100)
    plotter = Plotter("figures", file_name)
    writer = SummaryWriter(log_dir=f'runs/logs/{file_name}')
    plotter.set_capacity(suk.capacity)
    best_result = 0
    best_sol = np.array([])
    solution_list, solution_profit = [], []
    for _ in range(episodes):
        suk.reset()
        ga_solution, ga_fitness = ga_solver(
        suk,
        epochs=100,
        pop_size=50,
        pc=0.9,  # Crossover probability
        pm=0.4   # Mutation probability
        )
        solution_list.append(ga_solution)
        solution_profit.append(ga_fitness)
    avg_hamming = np.mean([[hamming_distance(sol1, sol2) / len(sol1) for sol2 in solution_list] for sol1 in solution_list])

    print(solution_profit)
    print(sum(solution_profit) / len(solution_profit))
    print(max(solution_profit))
    suk.set_init_sol(solution_list)
    print(f"Average normalized Hamming: {avg_hamming:.3f}")

    episode_sol = []
    episode_prof = []
    try:
        for e in range(episodes):
            print(f"Start episode {e + 1}")
            suk.reset_init(idx=e)
            state = suk.get_state()  # Assume env has reset/step
            terminate = False
            total_reward = 0.0
            loss = 0.0
            count = 0
            episode_entropy = []
            current_best_prof = suk.get_profit()
            current_best_sol = suk.get_state()

            while count < 100:
                count += 1
                action, entropy = agent.action(state)
                next_state, reward, terminate, success = suk.step(action)  # Adjust to your env's signature
                agent.decay_step()
                if success: 
                    agent.update_tabu(action)
                if np.isnan(reward):
                    raise ValueError(f"nan reward {reward}")
                agent.remember(state, action, reward, next_state, terminate)
                state = next_state
                total_reward += reward
                episode_entropy.append(entropy)
                if suk.get_profit() > current_best_prof:
                    current_best_prof = suk.get_profit()
                    current_best_sol = suk.get_state()
                if suk.get_profit() > best_result:
                    best_result = suk.get_profit()
                    best_sol = suk.get_state()
                loss += agent.replay(batch_size)
            episode_sol.append(current_best_sol)
            episode_prof.append(current_best_prof)
            loss = loss / count
            agent.reset_noise()
            agent.decay_episode()
            
            if (e + 1) % update_interval == 0:
                agent.update_target_model()
            writer.add_scalar('Reward/Episode', total_reward, e + 1)
            writer.add_scalar('Profit/Best', best_result, e + 1)
            writer.add_scalar('Loss/Average', loss, e + 1)
            writer.add_scalar('Weight/Final', suk.get_weight(), e + 1)
            writer.add_scalar('Entropy/Average', np.mean(episode_entropy), e + 1)
            
            print(f"Episode {e+1}, Reward: {total_reward}, Result: {current_best_prof}, Loss: {loss}, total step: {count}")
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("")
        print(f"Best result: {best_result}")
        print(f"Save result on result/{file_name}.npy")
        np.save(f"result/{file_name}.npy", best_sol)
        if save_checkpoint_:
            save_checkpoint(best_result, file_name, best_sol, agent)
        writer.close()
        return
    finally:
        result_str = ' '.join(['1' if x > 0.5 else '0' for x in best_sol])
        print(f"Result: {result_str}")
        suk.set_state(best_sol)
        print(f"Total weight: {suk.get_weight()}, capacity: {loader.capacity}")
        print(f"After ILS: Max Profit: {suk.get_profit}, Best sol: {best_sol}")
        print(f"Save result on result/{file_name}.npy")
        np.save(f"result/{file_name}.npy", best_sol)
        if save_checkpoint_:
            print(f"Save checkpoints in checkpoints/{file_name}.pth")
            save_checkpoint(best_result, file_name, best_sol, agent)
        writer.close()

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
