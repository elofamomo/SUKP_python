from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler


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
    i = 0
    terminate = True
    while terminate and i < handler.m:
        terminate = handler.add_item(after_sort_index[i])
        i += 1

    return handler.get_profit(), handler.get_weight()