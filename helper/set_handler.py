import numpy as np
import random
from mealpy import BinaryVar, Problem
from mealpy.evolutionary_based.GA import OriginalGA

class SetUnionHandler:
    def __init__(self, data, param):
        """
        Initializes the SetUnionHandler with data from SUKPLoader.
        
        :param data: Dictionary from SUKPLoader.get_data(), containing:
            - 'm': int, number of items
            - 'n': int, number of elements
            - 'capacity': float or int, knapsack capacity
            - 'item_profits': np.array[float], profits of items
            - 'element_weights': np.array[float], weights of elements
            - 'item_subsets': list[list[int]], subsets of element indices per item
        """
        self.m = data['m']
        self.n = data['n']
        self.capacity = float(data['capacity'])  # Ensure float for comparisons
        self.item_profits = data['item_profits']
        self.element_weights = data['element_weights']
        self.item_subsets = data['item_subsets']
        self.penalty = param['penalty']
        self.init_sol = []
        self.selected_items = set()
        self.element_counts = np.zeros(self.n, dtype=int)
        self.total_profit = 0.0
        self.total_weight = 0.0
        print("Terminate reward 0.0")
        self.terminate_reward = 0.0

    def add_item(self, item_idx):
        """
        Adds an item to the selection if not already selected.
        Updates total_profit and total_weight incrementally.
        
        :param item_idx: int, index of the item to add (0 to m-1)
        :return: bool, True if added (and feasible check can be done separately)
        """
        if item_idx in self.selected_items:
            return False
        
        additional_weight = 0.0
        for elem in self.item_subsets[item_idx]:
            if self.element_counts[elem] == 0:
                additional_weight += self.element_weights[elem]
        if self.total_weight + additional_weight > self.capacity:
            return False
        
        self.selected_items.add(item_idx)
        self.total_profit += self.item_profits[item_idx]
        for elem in self.item_subsets[item_idx]:
            prev_count = self.element_counts[elem]
            self.element_counts[elem] += 1
            if prev_count == 0:
                self.total_weight += self.element_weights[elem]
        
        return True

    def add_items(self, item_idxs):
        """
        Adds multiple items to the selection.
        
        :param item_idxs: list[int] or iterable of item indices
        :return: int, number of items actually added (skips duplicates)
        """
        added_count = 0
        for idx in item_idxs:
            if self.add_item(idx):
                added_count += 1
        return added_count

    def remove_item(self, item_idx):
        """
        Removes an item from the selection if selected.
        Updates total_profit and total_weight incrementally.
        
        :param item_idx: int, index of the item to remove
        :return: bool, True if removed
        """
        if item_idx not in self.selected_items:
            return False
        self.selected_items.remove(item_idx)
        self.total_profit -= self.item_profits[item_idx]
        for elem in self.item_subsets[item_idx]:
            self.element_counts[elem] -= 1
            if self.element_counts[elem] == 0:
                self.total_weight -= self.element_weights[elem]
        return True

    def get_profit(self):
        """
        Gets the current total profit (objective value).
        If the current total_weight > capacity, returns 0 (infeasible).
        Otherwise, returns the total_profit.
        
        :return: float, the value (profit if feasible, else 0)
        """
        return self.total_profit

    def get_totals(self):
        """
        Gets the raw totals without feasibility check.
        
        :return: tuple[float, float], (total_profit, total_weight)
        """
        return self.total_profit, self.total_weight
    
    def get_weight(self):
        return self.total_weight

    def is_feasible(self):
        """
        Checks if the current selection is feasible (total_weight <= capacity).
        
        :return: bool
        """
        return self.total_weight <= self.capacity

    def get_current_union(self):
        """
        Gets the current union of elements (indices where count > 0).
        
        :return: list[int], sorted list of unique element indices in the union
        """
        return np.where(self.element_counts > 0)[0].tolist()

    def reset(self):
        """
        Resets the selection to empty.
        """
        self.selected_items.clear()
        self.element_counts.fill(0)
        self.total_profit = 0.0
        self.total_weight = 0.0
    
    def reset_init(self):
        if not self.init_sol or len(self.init_sol) == 0:
            raise ValueError("Solution list is empty")
        idx = np.random.randint(0, len(self.init_sol))
        selected_solution = self.init_sol[idx]
        self.set_state(selected_solution)
    
    def set_init_sol(self, init_sol):
        if len(init_sol) > 0:
            self.init_sol = init_sol

    def set_state(self, solution):
        self.reset()
        selected_items = np.where(solution > 0.5)[0].tolist()
        self.add_items(selected_items)
    
    def get_profit_max(self):
        return max(self.item_profits)

    def get_profit_average(self):
        return max(self.item_profits) / len(self.item_profits)

    def get_state(self):
        return np.array([1.0 if i in self.selected_items else 0.0 for i in range(self.m)], dtype=float)
    
    def step(self, action):
        if action < 0 or action > 2 * self.m:
            raise ValueError(f"Action must be between 0 and {2 * self.m}. Current action is {action}")

        terminate = False
        reward = 0.0
        current_profit = self.get_profit()
        
        if action == 2 * self.m:		    	 	  
            reward = self.terminate_reward
            terminate = True
        else:
            if 0 <= action and action < self.m:
                added = self.add_item(action)
                if added:
                    reward = self.get_profit() - current_profit
                else:
                    reward = -self.penalty
            elif action > self.m and action <= 2 * self.m:
                removed = self.remove_item(action - self.m)
                if removed:
                    reward = self.get_profit() - current_profit
                else:
                    reward = -self.penalty
        
        new_state = self.get_state()
        return new_state, reward, terminate
    
    def iterated_local_search(self, solution_list, current_solution_list, current_profit_list, max_iter=500, tabu_size=20, perturbation_strength=3):
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
        for i in range(len(solution_list)):
            solution = solution_list[i]
            self.set_state(solution)
            best_solution = self.get_state().copy()
            best_profit = self.get_profit()
            current_profit = best_profit
            
            tabu_list = []  # List of tabu items (recently added/removed)
            
            for iter in range(max_iter):
                improved = False
                
                # Phase 1: Intensification (greedy neighborhood search)
                unselected = list(set(range(self.m)) - self.selected_items)
                selected = list(self.selected_items)
                
                # Try adding best non-tabu item by density
                if unselected:
                    densities = []
                    for item in unselected:
                        if item in tabu_list:
                            continue
                        add_weight = sum(self.element_weights[elem] for elem in self.item_subsets[item] if self.element_counts[elem] == 0)
                        if add_weight > 0 and self.total_weight + add_weight <= self.capacity:
                            density = self.item_profits[item] / add_weight
                            densities.append((item, density))
                    if densities:
                        best_add = max(densities, key=lambda x: x[1])[0]
                        self.add_item(best_add)
                        new_profit = self.get_profit()
                        if new_profit > current_profit:
                            current_profit = new_profit
                            improved = True
                            tabu_list.append(best_add)
                            if len(tabu_list) > tabu_size:
                                tabu_list.pop(0)
                        else:
                            self.remove_item(best_add)
                
                # Try removing worst non-tabu item (lowest profit contribution)
                if selected:
                    contributions = []
                    for item in selected:
                        if item in tabu_list:
                            continue
                        contributions.append((item, self.item_profits[item]))
                    if contributions:
                        worst_remove = min(contributions, key=lambda x: x[1])[0]
                        self.remove_item(worst_remove)
                        new_profit = self.get_profit()
                        if new_profit > current_profit:  # Rare, but possible if removal enables better adds later
                            current_profit = new_profit
                            improved = True
                            tabu_list.append(worst_remove)
                            if len(tabu_list) > tabu_size:
                                tabu_list.pop(0)
                        else:
                            self.add_item(worst_remove)
                
                # Update best if improved
                if current_profit > best_profit:
                    best_profit = current_profit
                    best_solution = self.get_state().copy()
                
                # Phase 2: Diversification (perturb if no improvement)
                if not improved and iter % 10 == 0:  # Perturb every 10 iterations if stuck
                    for _ in range(perturbation_strength):
                        if random.random() < 0.5 and selected:  # Remove random
                            item = random.choice(selected)
                            self.remove_item(item)
                        elif unselected:  # Add random if feasible
                            item = random.choice(unselected)
                            self.add_item(item)
                    current_profit = self.get_profit()
                if best_profit > current_profit_list[i]:
                    current_profit_list[i] = best_profit
                    current_solution_list[i] = best_solution
        overall_best_profit = max(current_profit_list)
        overall_best_solution = current_solution_list[current_profit_list.index(overall_best_profit)]
        return overall_best_solution, overall_best_profit
    """ 
    Time complexity : O(S * max_iter * m * n)
    S: Size of slution list
    max_iter: max iter
    m: self.m - number of items
    n: self.n - numer of elements
    """
        

