import random

def ils_with_tabu(handler, start_solution, max_iter=100, tabu_tenure=10, perturb_strength=3):
    """
    ILS with tabu search to refine a solution.
    
    :param handler: SetUnionHandler instance
    :param start_solution: np.array, initial binary solution
    :param max_iter: int, max iterations
    :param tabu_tenure: int, steps an item is tabu
    :param perturb_strength: int, flips in perturbation
    :return: np.array, float; best solution and profit
    """
    handler.set_state(start_solution)
    current_solution = handler.get_state().copy()
    current_profit = handler.get_profit()
    best_solution = current_solution.copy()
    best_profit = current_profit
    tabu_list = {}  # item_idx: tenure
    no_improve_count = 0
    
    for _ in range(max_iter):
        improved = False
        unselected = [i for i in range(handler.m) if i not in handler.selected_items and (i not in tabu_list or tabu_list.get(i, 0) <= 0)]
        selected = [i for i in handler.selected_items if i not in tabu_list or tabu_list.get(i, 0) <= 0]
        
        # Add best by densities
        if unselected:
            densities = []
            for item in unselected:
                marginal_weight = sum(handler.element_weights[e] for e in handler.item_subsets[item] if handler.element_counts[e] == 0)
                if marginal_weight > 0 and handler.total_weight + marginal_weight <= handler.capacity:
                    densities.append((item, handler.item_profits[item] / marginal_weight))
            if densities:
                best_add, _ = max(densities, key=lambda x: x[1])
                handler.add_item(best_add)
                new_profit = handler.get_profit()
                if new_profit > current_profit:
                    current_profit = new_profit
                    improved = True
                    tabu_list[best_add] = tabu_tenure
                else:
                    handler.remove_item(best_add)
        
        # Remove worst to enable better add
        if selected and not improved:
            profits = [(item, handler.item_profits[item]) for item in selected]
            worst_remove, _ = min(profits, key=lambda x: x[1])
            handler.remove_item(worst_remove)
            unselected.append(worst_remove)
            densities = []
            for item in unselected:
                marginal_weight = sum(handler.element_weights[e] for e in handler.item_subsets[item] if handler.element_counts[e] == 0)
                if marginal_weight > 0 and handler.total_weight + marginal_weight <= handler.capacity:
                    densities.append((item, handler.item_profits[item] / marginal_weight))
            if densities:
                best_add, _ = max(densities, key=lambda x: x[1])
                handler.add_item(best_add)
                new_profit = handler.get_profit()
                if new_profit > current_profit:
                    current_profit = new_profit
                    improved = True
                    tabu_list[best_add] = tabu_tenure
                else:
                    handler.remove_item(best_add)
                    handler.add_item(worst_remove)
            else:
                handler.add_item(worst_remove)
        
        # Update best
        if current_profit > best_profit:
            best_profit = current_profit
            best_solution = handler.get_state().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Perturb if stuck
        if no_improve_count >= 10:
            for _ in range(perturb_strength):
                if random.random() < 0.5 and handler.selected_items:
                    item = random.choice(list(handler.selected_items))
                    handler.remove_item(item)
                elif unselected:
                    item = random.choice(unselected)
                    marginal_weight = sum(handler.element_weights[e] for e in handler.item_subsets[item] if handler.element_counts[e] == 0)
                    if handler.total_weight + marginal_weight <= handler.capacity:
                        handler.add_item(item)
            current_profit = handler.get_profit()
            no_improve_count = 0
        
        # Decay tabu
        for item in list(tabu_list):
            tabu_list[item] -= 1
            if tabu_list[item] <= 0:
                del tabu_list[item]
    
    return best_solution, best_profit