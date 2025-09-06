from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from solver.epsilon_greedy_agent import DQNAgent
import numpy as np
from mealpy import BinaryVar, Problem
from mealpy.evolutionary_based.GA import OriginalGA
from mealpy.swarm_based.ABC import OriginalABC
import torch
import torch.nn as nn


def greedy_init(handler: SetUnionHandler):
    """
    Generates an initial solution using a greedy approach.
    Iteratively adds the item with the highest profit / additional_weight density,
    prioritizing items with zero additional weight and positive profit.
    
    :param handler: SetUnionHandler instance
    :return: np.array[float], binary state vector (1.0 if selected, 0.0 else)
    """
    handler.reset()
    unselected = set(range(handler.m))
    
    while True:
        best_item = None
        best_density = -float('inf')
        best_add_weight = 0.0
        
        for item in list(unselected):  # Copy to avoid modification during iteration
            add_weight = 0.0
            for elem in handler.item_subsets[item]:
                if handler.element_counts[elem] == 0:
                    add_weight += handler.element_weights[elem]
            
            if handler.total_weight + add_weight > handler.capacity:
                continue  # Skip if it would exceed capacity
            
            if add_weight == 0:
                if handler.item_profits[item] > 0:
                    # Add immediately if free profit
                    handler.add_item(item)
                    unselected.remove(item)
                    best_item = None  # Reset to continue loop
                    break
                else:
                    continue
            
            density = handler.item_profits[item] / add_weight if add_weight > 0 else -float('inf')
            
            if density > best_density:
                best_density = density
                best_item = item
                best_add_weight = add_weight
        
        if best_item is None:
            break
        
        handler.add_item(best_item)
        unselected.remove(best_item)
    
    return handler.get_profit(), handler.get_weight()

def main():
    yaml_path = "helper/config.yaml"
    loader = SUKPLoader(yaml_path)
    data = loader.get_data()
    param = loader.get_param()
    suk = SetUnionHandler(data, param)
    profit, weight = greedy_init(suk)
    print(profit)
    print(weight)
    m = nn.Softmax(dim = 1)
    input = torch.randn(1, 3)
    print(input)
    output = m(input)
    print(output)


if __name__ == "__main__":
    main()