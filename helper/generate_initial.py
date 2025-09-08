from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from solver.epsilon_greedy_agent import DQNAgent
import numpy as np
from mealpy import BinaryVar, Problem
from mealpy.evolutionary_based.GA import OriginalGA
from mealpy.swarm_based.ABC import OriginalABC


class SUKPProblem(Problem):
    """
    Custom Problem class for Set Union Knapsack Problem in Mealpy.
    Uses BinaryVar for binary solutions.
    """
    def __init__(self, handler: SetUnionHandler, *args, **kwargs):
        bounds = BinaryVar(n_vars=handler.m)
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

    init_pop = None
    if init_solution is not None:
        init_pop = [init_solution] + [np.random.randint(0, 2, size=handler.m).astype(float) for _ in range(pop_size - 1)]

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

    model.solve(problem)
    
    best_solution = model.g_best.solution  # Ensure binary (0 or 1)
    best_fitness = model.g_best.target.fitness
    
    # Verify solution feasibility
    handler.reset()
    selected = np.where(best_solution > 0.5)[0].tolist()
    handler.add_items(selected)
    if not handler.is_feasible():
        print("Warning: GA solution is infeasible.")
        best_profit = 0.0
    
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

def ga_gen_init_set(suk: SetUnionHandler):
    solution_list = []
    solution_profit = []
    for i in range(100):
        suk.reset()
        ga_solution, ga_fitness = ga_solver(
        suk,
        epochs=100,
        pop_size=50,
        init_solution=greedy_init1,
        pc=0.9,  # Crossover probability
        pm=0.1   # Mutation probability
        )
        solution_list.append(ga_solution)
        solution_profit.append(ga_fitness)
    print(solution_profit)
    return solution_list