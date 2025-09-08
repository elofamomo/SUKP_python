import numpy as np

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
    
    def reset_init(self, solution_list):
        if not solution_list:
            raise ValueError("Solution list is empty")
        idx = np.random.randint(0, len(solution_list))
        selected_solution = solution_list[idx]
        self.reset()
        selected_items = np.where(selected_solution > 0.5)[0].tolist()
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
    