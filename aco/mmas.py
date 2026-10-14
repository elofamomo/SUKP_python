import numpy as np
from aco.instance import Cover
from aco.sgroa import ratio


def construct(inst, tau_a, eta_static, beta, rng):
    """
    One ant: start from an empty knapsack and repeatedly pick one item among those
    that still fit, with probability proportional to tau_i^alpha * eta_i^beta, until
    nothing fits. The result is always feasible and maximal (no item can be added),
    so S-GROA's repairing stage is never needed and its optimizing stage is built in.

    Items whose elements are all covered already add no weight and are taken at
    once; this does not change the distribution of the final set.

    :param tau_a: tau^alpha, np.array (m,)
    :param eta_static: p_i / R_i, or None for the dynamic heuristic p_i / gain_i
    :return: Cover
    """
    cover = Cover(inst)
    attract = tau_a * eta_static ** beta if eta_static is not None else None
    while True:
        cand = ~cover.selected & (cover.gain <= inst.capacity - cover.load)
        free = cand & (cover.gain == 0)
        if free.any():
            for i in np.flatnonzero(free):
                cover.add(i)
            cand &= ~free
        idx = np.flatnonzero(cand)
        if idx.size == 0:
            return cover
        if attract is not None:
            w = attract[idx]
        else:
            w = tau_a[idx] * (inst.profit[idx] / cover.gain[idx]) ** beta
        c = np.cumsum(w)
        if c[-1] > 0:
            pick = np.searchsorted(c, rng.random() * c[-1], side='right')
        else:
            pick = rng.integers(idx.size)
        cover.add(idx[pick])


def mmas(inst, n_ants=20, max_iter=None, alpha=1.0, beta=2.0, rho=0.1, tau_min=0.01,
         heuristic='static', rng=None):
    """
    MAX-MIN Ant System for SUKP with one pheromone value per item.

    A solution is a set, so the pheromone sits on items (as in ACO for other
    subset problems), not on pairs of consecutive choices. After each iteration
    the iteration-best ant reinforces its items (hyper-cube update):
        tau <- (1 - rho) * tau + rho * [i in best],  clamped to [tau_min, 1],
    starting from tau = 1 everywhere.

    :param inst: Instance
    :param n_ants: ants per iteration (20 = BABC's population, for the same budget)
    :param max_iter: iterations (default max(m, n), as BABC)
    :param alpha: pheromone exponent
    :param beta: heuristic exponent
    :param rho: evaporation rate
    :param tau_min: lower pheromone bound (the upper bound is 1)
    :param heuristic: 'static' eta_i = p_i / R_i (S-GROA's ratio) or
                      'dynamic' eta_i = p_i / gain_i (weight item i adds to the current knapsack)
    :param rng: np.random.Generator
    :return: dict with 'solution' (bool np.array), 'profit', 'history' (best profit
             after each iteration) and 'evaluations' (ants constructed)
    """
    if heuristic not in ('static', 'dynamic'):
        raise ValueError("heuristic must be 'static' or 'dynamic'")
    rng = rng or np.random.default_rng()
    max_iter = max_iter or max(inst.m, inst.n)
    eta_static = ratio(inst) if heuristic == 'static' else None
    if eta_static is not None and np.isinf(eta_static).any():
        # an item with no element is free; Cover takes it at once, its eta never matters
        eta_static = np.where(np.isinf(eta_static), 0.0, eta_static)
    tau = np.ones(inst.m)

    best_y, best_f = None, -1.0
    history = []
    for _ in range(max_iter):
        tau_a = tau ** alpha
        it_best = None
        for _ in range(n_ants):
            cover = construct(inst, tau_a, eta_static, beta, rng)
            if it_best is None or cover.profit > it_best.profit:
                it_best = cover
        if it_best.profit > best_f:
            best_y, best_f = it_best.selected, it_best.profit
        tau = np.clip((1 - rho) * tau + rho * it_best.selected, tau_min, 1.0)
        history.append(best_f)

    return {'solution': best_y, 'profit': best_f, 'history': history,
            'evaluations': n_ants * max_iter}
