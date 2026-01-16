from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from solver.epsilon_greedy_agent import DQNAgent
import numpy as np
from mealpy import BinaryVar, Problem
from mealpy.evolutionary_based.GA import OriginalGA
from mealpy.swarm_based.ABC import OriginalABC
import random
import os
from helper.k_heapq import TopKHeap
from helper.generate_initial import random_gen

class SUKPProblem(Problem):
    """
    Custom Problem class for Set Union Knapsack Problem in Mealpy.
    Uses BinaryVar for binary solutions.
    """
    def __init__(self, handler: SetUnionHandler, *args, **kwargs):
        bounds = BinaryVar(n_vars=handler.m)
        kwargs['log_to'] = None
        super().__init__(bounds=bounds, minmax="max", *args, **kwargs)
        self.handler = handler
        
    
    def obj_func(self, solution):
        """
        Objective function: Returns profit if feasible, else 0.
        """
        self.handler.reset()
        selected = np.where(solution > 0.5)[0].tolist()  # Threshold for binary
        self.handler.add_items(selected)
        return self.handler.get_profit()

def ga_solver(handler, epochs = 100, pop_size = 50, init_solution = None, pc= 0.9, pm=0.1):
    problem = SUKPProblem(handler)

    # init_pop = None
    # if init_solution is not None:
    #     init_pop = [init_solution] + [np.random.randint(0, 2, size=handler.m).astype(float) for _ in range(pop_size - 1)]
    
    # Configure GA parameters
    model = OriginalGA(
        epoch=epochs,
        pop_size=pop_size,
        pc=pc,  # Crossover probability
        pm=pm,  # Mutation probability
        selection="tournament",  # Tournament selection for better convergence
        k_tournament=4,  # Size of tournament
        crossover="uniform",  # Uniform crossover for binary problems
        mutation="flip"  # Bit-flip mutation for binary variables
    )
    if init_solution is None:
        model.solve(problem)
    else:
        model.solve(problem,starting_solutions=init_solution)
    
    best_solution = model.g_best.solution  # Ensure binary (0 or 1)
    best_fitness = model.g_best.target.fitness
    
    return best_solution, best_fitness


def greedy_init1(handler: SetUnionHandler):
    handler.reset()
    items = [i for i in range(handler.m)]
    dense = []
    for item in range(handler.m):
        weight = 0
        for elem in handler.item_subsets[item]:
            weight += handler.element_weights[elem]
        dense.append(handler.item_profits[item] / weight)
    combined = sorted(zip(dense, items),reverse=True)
    after_sort_dense, after_sort_index = zip(*combined)
    after_sort_index = list(after_sort_index)
    print(len(after_sort_index))
    i = 0
    terminate = True
    while terminate and i < handler.m:
        terminate = handler.add_item(after_sort_index[i])
        i += 1

    return handler.get_profit(), handler.get_weight()

def implement_greedy(func, handler):
    return func(handler)

def iterated_local_search(handler: SetUnionHandler, solution_list, max_iter=500, tabu_size=20, perturbation_strength=3):
    """
    Performs Iterated Local Search starting from a random top GA solution.
    :param handler: SetUnionHandler instance
    :param solution_list: list[np.array], top binary solutions from GA
    :param max_iter: int, maximum iterations
    :param tabu_size: int, size of tabu list (forbidden moves)
    :param perturbation_strength: int, number of random changes in diversification
    :return: np.array, best solution; float, best profit
    """
    if not solution_list:
        raise ValueError("Solution list is empty")
    
    # Initialize from a random GA solution
    handler.reset_init(solution_list)
    best_solution = handler.get_state().copy()
    best_profit = handler.get_profit()
    current_profit = best_profit
    
    tabu_list = []  # List of tabu items (recently added/removed)
    
    for iter in range(max_iter):
        improved = False
        
        # Phase 1: Intensification (greedy neighborhood search)
        unselected = list(set(range(handler.m)) - handler.selected_items)
        selected = list(handler.selected_items)
        
        # Try adding best non-tabu item by density
        if unselected:
            densities = []
            for item in unselected:
                if item in tabu_list:
                    continue
                add_weight = sum(handler.element_weights[elem] for elem in handler.item_subsets[item] if handler.element_counts[elem] == 0)
                if add_weight > 0 and handler.total_weight + add_weight <= handler.capacity:
                    density = handler.item_profits[item] / add_weight
                    densities.append((item, density))
            if densities:
                best_add = max(densities, key=lambda x: x[1])[0]
                handler.add_item(best_add)
                new_profit = handler.get_profit()
                if new_profit > current_profit:
                    current_profit = new_profit
                    improved = True
                    tabu_list.append(best_add)
                    if len(tabu_list) > tabu_size:
                        tabu_list.pop(0)
                else:
                    handler.remove_item(best_add)
        
        # Try removing worst non-tabu item (lowest profit contribution)
        if selected:
            contributions = []
            for item in selected:
                if item in tabu_list:
                    continue
                contributions.append((item, handler.item_profits[item]))
            if contributions:
                worst_remove = min(contributions, key=lambda x: x[1])[0]
                handler.remove_item(worst_remove)
                new_profit = handler.get_profit()
                if new_profit > current_profit:  # Rare, but possible if removal enables better adds later
                    current_profit = new_profit
                    improved = True
                    tabu_list.append(worst_remove)
                    if len(tabu_list) > tabu_size:
                        tabu_list.pop(0)
                else:
                    handler.add_item(worst_remove)
        
        # Update best if improved
        if current_profit > best_profit:
            best_profit = current_profit
            best_solution = handler.get_state().copy()
        
        # Phase 2: Diversification (perturb if no improvement)
        if not improved and iter % 10 == 0:  # Perturb every 10 iterations if stuck
            for _ in range(perturbation_strength):
                if random.random() < 0.5 and selected:  # Remove random
                    item = random.choice(selected)
                    handler.remove_item(item)
                elif unselected:  # Add random if feasible
                    item = random.choice(unselected)
                    handler.add_item(item)
            current_profit = handler.get_profit()
    
    return best_solution, best_profit

def tabu_search(handler, solution_list, max_iter=500, tabu_size=10):
    """
    Performs Tabu Search starting from a random top GA solution.
    :param handler: SetUnionHandler instance
    :param solution_list: list[np.array], top binary solutions from GA
    :param max_iter: int, maximum iterations
    :param tabu_size: int, size of tabu list (forbidden moves)
    :return: np.array, best solution; float, best profit
    """
    if not solution_list:
        raise ValueError("Solution list is empty")
    
    # Initialize from a random GA solution
    handler.reset_init(solution_list)
    best_solution = handler.get_state().copy()
    best_profit = handler.get_profit()
    current_profit = best_profit
    
    tabu_list = []  # List of (item, action) tuples
    max_tabu_size = tabu_size
    
    for _ in range(max_iter):
        best_neighbor = None
        best_neighbor_profit = -float('inf')
        best_move = None
        
        # Generate neighborhood: add/remove one item
        unselected = list(set(range(handler.m)) - handler.selected_items)
        selected = list(handler.selected_items)
        
        # Try adding items
        for item in unselected:
            if (item, 'add') in tabu_list:
                continue
            handler.add_item(item)
            profit = handler.get_profit()
            if profit > best_neighbor_profit:
                best_neighbor_profit = profit
                best_neighbor = handler.get_state().copy()
                best_move = (item, 'add')
            handler.remove_item(item)  # Undo
            
        # Try removing items
        for item in selected:
            if (item, 'remove') in tabu_list:
                continue
            handler.remove_item(item)
            profit = handler.get_profit()
            if profit > best_neighbor_profit:
                best_neighbor_profit = profit
                best_neighbor = handler.get_state().copy()
                best_move = (item, 'remove')
            handler.add_item(item)  # Undo
        
        # Apply best non-tabu move
        if best_neighbor is not None:
            handler.reset()
            handler.add_items(np.where(best_neighbor > 0.5)[0].tolist())
            current_profit = best_neighbor_profit
            tabu_list.append(best_move)
            if len(tabu_list) > max_tabu_size:
                tabu_list.pop(0)
            
            # Update best if improved
            if current_profit > best_profit:
                best_profit = current_profit
                best_solution = best_neighbor.copy()
        
        # Aspiration: Override tabu if significantly better than best_profit
        for item in unselected:
            if (item, 'add') in tabu_list and current_profit < best_profit * 1.1:  # Allow 10% better
                handler.add_item(item)
                profit = handler.get_profit()
                if profit > best_profit:
                    best_profit = profit
                    best_solution = handler.get_state().copy()
                    tabu_list.append((item, 'add'))
                    if len(tabu_list) > max_tabu_size:
                        tabu_list.pop(0)
                handler.remove_item(item)
        
        for item in selected:
            if (item, 'remove') in tabu_list and current_profit < best_profit * 1.1:
                handler.remove_item(item)
                profit = handler.get_profit()
                if profit > best_profit:
                    best_profit = profit
                    best_solution = handler.get_state().copy()
                    tabu_list.append((item, 'remove'))
                    if len(tabu_list) > max_tabu_size:
                        tabu_list.pop(0)
                handler.add_item(item)
    
    return best_solution, best_profit


def main():
    yaml_path = "helper/config.yaml"
    loader = SUKPLoader(yaml_path)
    data = loader.get_data()
    param = loader.get_param()
    suk = SetUnionHandler(data, param)
    result_dir = 'result'
    file_name = loader.get_filename()
    npy_path = os.path.join(result_dir, f'{file_name}.npy')
    
    if not os.path.exists(npy_path):
        print(f"File {npy_path} does not exist.")
        return
    
    best_sol = np.load(npy_path)
    
    if len(best_sol) != data['m']:
        print("Loaded best_sol has incorrect length.")
        return
    
    
    # Add items based on best_sol
    for i in range(len(best_sol)):
        if best_sol[i] > 0.5:  # Since it's floats 0.0 or 1.0
            suk.add_item(i)
    
    result_str = ' '.join(['1' if x > 0.5 else '0' for x in best_sol])
    print(f"Result: {result_str}")
    print(f"Total weight: {suk.get_weight()}, capacity: {loader.capacity}")
    
    # Get and print total profit
    total_profit = suk.get_profit()
    print(f"Total profit: {total_profit}")
    solution_list = random_gen(suk)
    solution_list = []
    solution_profit = []
    # for _ in range(100):
    #     suk.reset()
    #     ga_solution, ga_fitness = ga_solver(
    #     suk,
    #     epochs=100,
    #     pop_size=1000,
    #     pc=0.9,  # Crossover probability
    #     pm=0.1   # Mutation probability
    #     )
    #     solution_list.append(ga_solution)
    #     solution_profit.append(ga_fitness)
    # print(solution_profit)
    # print(sum(solution_profit) / len(solution_profit))
    # print(max(solution_profit))
    heap = TopKHeap(100)
    solution_list = [best_sol]
    solution_profit = [total_profit]
    print(solution_profit)
    for i in range(10):
        current_solution_list = solution_list.copy()  # Run 10 times for diversity
        current_profit_list = solution_profit.copy()
        for _ in range(200): 
            current_best_sol, current_best_prof = suk.iterated_local_search(solution_list, current_solution_list, current_profit_list)
        print(f"Loop {i}: Current profit: {current_profit_list}, \n Best profit: {current_best_prof}")
        if max(current_profit_list) > max(solution_profit):
            best_sol = current_best_sol
            solution_list = current_solution_list
            solution_profit = current_profit_list
            

    max_profit = max(solution_profit)
    best_sol = solution_list[solution_profit.index(max_profit)]
    result_str = ' '.join(['1' if x > 0.5 else '0' for x in best_sol])
    print(f"Result: {result_str}")
    print(f"Total weight: {suk.get_weight()}, capacity: {loader.capacity}")
    np.save(f"result/{file_name}.npy", best_sol)
    avg_profit = np.mean(solution_profit)

    print(f"After ILS: Max Profit: {max_profit}, Average Profit: {avg_profit}")

    
if __name__ == "__main__":
    main()
    