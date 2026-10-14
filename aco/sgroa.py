import numpy as np
from aco.instance import Cover


def ratio(inst):
    """
    Greedy metric of He et al. (2018), Sec. 4.1: p_i / R_i with
    R_i = sum_{j in U_i} w_j / d_j and d_j the number of items containing element j,
    so the weight of a shared element is split evenly between the items that hold it.

    :return: np.array (m,), inf for an item that covers no element
    """
    freq = inst.relation.sum(axis=0)
    share = np.divide(inst.weight, freq, out=np.zeros(inst.n), where=freq > 0)
    r = inst.relation @ share
    return np.divide(inst.profit, r, out=np.full(inst.m, np.inf), where=r > 0)


def ratio_order(inst):
    """H[1..m]: item indices sorted by p_i / R_i in descending order (Eq. (10))."""
    return np.argsort(-ratio(inst), kind='stable')


def sgroa(inst, order, y):
    """
    S-GROA (Algorithm 2): greedy repairing and optimization.

    Repairing stage, only if y is infeasible: rebuild from an empty knapsack,
    walking H and keeping each item of y that still fits.
    Optimizing stage: walk H again and add every item that still fits.

    :param inst: Instance
    :param order: H, from ratio_order(inst)
    :param y: bool np.array (m,), the potential solution
    :return: Cover holding the feasible solution and its profit f(Y)
    """
    order = order.tolist()
    if inst.union_weight(y) <= inst.capacity:
        cover = Cover.from_items(inst, y)
    else:
        cover = Cover(inst)
        for i in order:
            if y[i] and cover.fits(i):
                cover.add(i)
    for i in order:
        if not cover.selected[i] and cover.fits(i):
            cover.add(i)
    return cover
