import numpy as np
from aco.sgroa import ratio_order, sgroa


def babc(inst, n_bees=20, max_iter=None, a=5.0, limit=None, rng=None):
    """
    Binary artificial bee colony for SUKP (He et al. 2018, Algorithm 3).

    A food source is a real vector X in [-a, a]^m. It maps to an item selection
    by sign, y_i = 1 iff x_i >= 0 (Eq. (9)), which S-GROA then repairs and fills;
    the fitness of X is the profit of that feasible solution. The paper leaves
    two details open, decided here as follows:
      * Eq. (6) moves every dimension, each with its own phi in [-1, 1] (the
        "full mapping"), and V is clipped back into [-a, a]^m.
      * The onlooker phase follows Algorithm 3 step 4.2 literally: bee i searches
        once if p_i > rand(0, 1), with p_i = fit_i / sum(fit) (Eq. (7)).

    :param inst: Instance
    :param n_bees: N, population size (paper: 20)
    :param max_iter: MIT (paper: max(m, n))
    :param a: bound of the real search space (paper: 5.0)
    :param limit: trials without improvement before a scout restart (paper: max(m, n) / 5)
    :param rng: np.random.Generator
    :return: dict with 'solution' (bool np.array), 'profit', 'history' (best profit
             after each iteration) and 'evaluations' (S-GROA calls)
    """
    rng = rng or np.random.default_rng()
    m = inst.m
    max_iter = max_iter or max(inst.m, inst.n)
    limit = limit or max(inst.m, inst.n) / 5
    order = ratio_order(inst)
    evaluations = 0

    def evaluate(x):
        nonlocal evaluations
        evaluations += 1
        cover = sgroa(inst, order, x >= 0)
        return cover.selected, cover.profit

    X = rng.uniform(-a, a, size=(n_bees, m))  # Eq. (8)
    fit = np.zeros(n_bees)
    best_y, best_f = None, -1.0
    for i in range(n_bees):
        y, fit[i] = evaluate(X[i])
        if fit[i] > best_f:
            best_y, best_f = y, fit[i]
    trial = np.zeros(n_bees, dtype=int)

    def search(i):
        """Eq. (6) around food source i with a random partner k != i, then greedy selection."""
        nonlocal best_y, best_f
        k = rng.integers(n_bees - 1)
        k += k >= i
        v = np.clip(X[i] + rng.uniform(-1, 1, m) * (X[i] - X[k]), -a, a)
        y, f = evaluate(v)
        if f > fit[i]:
            X[i], fit[i], trial[i] = v, f, 0
            if f > best_f:
                best_y, best_f = y, f
        else:
            trial[i] += 1

    history = []
    for _ in range(max_iter):
        for i in range(n_bees):  # employed bees
            search(i)
        p = fit / fit.sum()  # Eq. (7)
        for i in range(n_bees):  # onlooker bees
            if p[i] > rng.random():
                search(i)
        for i in range(n_bees):  # scout bees
            if trial[i] > limit:
                X[i] = rng.uniform(-a, a, m)
                y, fit[i] = evaluate(X[i])
                trial[i] = 0
                if fit[i] > best_f:
                    best_y, best_f = y, fit[i]
        history.append(best_f)

    return {'solution': best_y, 'profit': best_f, 'history': history, 'evaluations': evaluations}
