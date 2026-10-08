import numpy as np


def generate_instance(m, n, alpha, beta, rng, max_value=500):
    """
    Random SUKP instance shaped like the benchmark sets: item i covers element j with probability
    alpha, capacity is beta times the total element weight, and profits and weights are uniform
    integers in [1, max_value] (as in Set II).

    :param rng: np.random.Generator
    :return: tuple (profits np.array[int64] (m), weights np.array[int64] (n), subsets list[np.array[int64]], capacity int)
    """
    profits = rng.integers(1, max_value + 1, size=m)
    weights = rng.integers(1, max_value + 1, size=n)
    relation = rng.random((m, n)) < alpha
    # Every benchmark item covers at least one element
    empty = np.flatnonzero(~relation.any(axis=1))
    relation[empty, rng.integers(0, n, size=len(empty))] = True
    subsets = [np.flatnonzero(row) for row in relation]
    capacity = int(beta * weights.sum())
    return profits, weights, subsets, capacity
