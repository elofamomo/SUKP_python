from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from helper.k_heapq import TopKHeap
from helper.generate_initial import ga_solver, hamming_distance
from solver.softmax_agent import DQNAgent
import torch
import numpy as np
from networks.dqn100 import DeepQlearningNetwork


def run():
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
    episodes = config['episodes']
    file_name = loader.get_filename()
    suk = SetUnionHandler(data, param)
    agent = DQNAgent(suk, device, load_checkpoint=True, file_name=file_name)
    agent.model.eval()
    solution_list, solution_profit = [], []
    best_result = 0
    best_sol = np.array([])
    num_of_solut = 200    
    max_steps = 500
    for _ in range(num_of_solut):
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
    suk.set_init_sol(solution_list) 
    print(f"Average normalized Hamming: {avg_hamming:.3f}")
    try:
        for solut in range(num_of_solut):
            suk.reset_init(idx=solut)
            state = suk.get_state()
            current_best_prof = suk.get_profit()
            current_best_sol = suk.get_state()
            steps = 0
            while steps < max_steps:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device)
                

    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("")
        print(f"Best result: {best_result}")
        print(f"Save result on result/{file_name}.npy")
        np.save(f"result/{file_name}.npy", best_sol)        
    finally:
        print("")
        print(f"Best result: {best_result}")
        print(f"Save result on result/{file_name}.npy")
        np.save(f"result/{file_name}.npy", best_sol) 


if __name__ == "__main__":
      run()