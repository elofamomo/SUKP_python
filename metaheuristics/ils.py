import random
import numpy as np
from helper.set_handler import SetUnionHandler


def ils_with_tabu(handler: SetUnionHandler, start_solution, max_iter=300, tabu_tenure=15, perturb_strength=3):
    """
    Tabu search with single-add, single-remove, and 1-1 swap moves.

    Swap moves (remove A, add B atomically) escape capacity-constrained local
    optima where no single add is feasible but a trade is. Each iteration
    evaluates the full neighborhood and picks the best non-tabu move (best-
    improvement strategy). Tabu tracks the reverse of each applied move to
    prevent immediate cycling. Aspiration: a tabu move is accepted if it
    strictly beats the global best.

    :param handler: SetUnionHandler (modified in-place; state set to best on return)
    :param start_solution: np.array binary solution to start from
    :param max_iter: tabu search iterations
    :param tabu_tenure: iterations the reverse of a move stays forbidden
    :param perturb_strength: items randomly flipped when stuck for 20 iterations
    :return: (best_solution np.array, best_profit float)
    """
    handler.set_state(start_solution)
    best_solution = handler.get_state().copy()
    best_profit = handler.get_profit()
    current_profit = best_profit

    tabu: dict = {}  # move_tuple -> remaining tenure
    no_improve = 0

    for _ in range(max_iter):
        best_move = None
        best_move_profit = float('-inf')
        best_move_state = None

        unselected = [i for i in range(handler.m) if i not in handler.selected_items]
        selected = list(handler.selected_items)

        # Single add
        for item in unselected:
            move = ('add', item)
            if handler.add_item(item):
                profit = handler.get_profit()
                if (move not in tabu or profit > best_profit) and profit > best_move_profit:
                    best_move_profit = profit
                    best_move_state = handler.get_state().copy()
                    best_move = move
                handler.remove_item(item)

        # Single remove
        for item in selected:
            move = ('remove', item)
            handler.remove_item(item)
            profit = handler.get_profit()
            if (move not in tabu or profit > best_profit) and profit > best_move_profit:
                best_move_profit = profit
                best_move_state = handler.get_state().copy()
                best_move = move
            handler.add_item(item)

        # 1-1 swap: remove a, add b atomically.
        # a is removed once per outer loop, all b candidates tried, then a restored.
        for a in selected:
            handler.remove_item(a)
            for b in unselected:
                move = ('swap', a, b)
                if handler.add_item(b):
                    profit = handler.get_profit()
                    if (move not in tabu or profit > best_profit) and profit > best_move_profit:
                        best_move_profit = profit
                        best_move_state = handler.get_state().copy()
                        best_move = move
                    handler.remove_item(b)
            handler.add_item(a)

        if best_move_state is None:
            break

        handler.set_state(best_move_state)
        current_profit = best_move_profit

        # Tabu the reverse move to prevent cycling back
        if best_move[0] == 'add':
            tabu[('remove', best_move[1])] = tabu_tenure
        elif best_move[0] == 'remove':
            tabu[('add', best_move[1])] = tabu_tenure
        else:  # swap (a, b) -> reverse is swap (b, a)
            tabu[('swap', best_move[2], best_move[1])] = tabu_tenure

        if current_profit > best_profit:
            best_profit = current_profit
            best_solution = best_move_state.copy()
            no_improve = 0
        else:
            no_improve += 1

        # Perturbation when stuck
        if no_improve >= 20:
            for _ in range(perturb_strength):
                if random.random() < 0.5 and handler.selected_items:
                    handler.remove_item(random.choice(list(handler.selected_items)))
                else:
                    candidates = [i for i in range(handler.m) if i not in handler.selected_items]
                    if candidates:
                        item = random.choice(candidates)
                        marginal = sum(
                            handler.element_weights[e]
                            for e in handler.item_subsets[item]
                            if handler.element_counts[e] == 0
                        )
                        if handler.total_weight + marginal <= handler.capacity:
                            handler.add_item(item)
            current_profit = handler.get_profit()
            no_improve = 0

        # Decay tabu tenures
        for move in list(tabu):
            tabu[move] -= 1
            if tabu[move] <= 0:
                del tabu[move]

    handler.set_state(best_solution)
    return best_solution, best_profit
