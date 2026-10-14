import re
import os
import numpy as np


class Instance:
    """
    One SUKP instance: m items, n elements, item i has profit[i] and covers the
    elements subsets[i], element j has weight[j]. A set of items is feasible when
    the weight of the union of their elements is <= capacity (He et al. 2018, Sec. 2).
    """

    def __init__(self, name, profit, weight, relation, capacity):
        self.name = name
        self.m, self.n = relation.shape
        self.capacity = float(capacity)
        # Profits and weights are integers, so float sums stay exact and can use BLAS
        self.profit = np.asarray(profit, dtype=float)
        self.weight = np.asarray(weight, dtype=float)
        self.relation = np.asarray(relation, dtype=bool)  # (m, n)
        self.subsets = [np.flatnonzero(row) for row in self.relation]
        self.item_weight = self.relation @ self.weight    # weight of each item alone
        # Cover.add gathers rows of this matrix on every add; float32 halves that traffic and
        # stays exact while the total element weight is below 2^24 (Set I/II: at most 254924)
        if self.weight.sum() >= 2 ** 24:
            raise ValueError(f"{name}: total element weight too large for float32 gain updates")
        self.relation_t = np.ascontiguousarray(self.relation.T, dtype=np.float32)  # (n, m)
        self.weight32 = self.weight.astype(np.float32)

    @classmethod
    def from_file(cls, path):
        """
        Parses the benchmark format of SUKP_instances_60 (header line with m, n and
        the knapsack size, then the profits, the weights and the m x n relation matrix).
        """
        with open(path) as f:
            text = f.read()
        m = int(re.search(r'm=(\d+)', text).group(1))
        n = int(re.search(r'n=(\d+)', text).group(1))
        capacity = int(re.search(r'size=(\d+)', text).group(1))
        _, rest = text.split('The profit of', 1)
        profit_part, rest = rest.split('The weight of', 1)
        weight_part, matrix_part = rest.split('Relation matrix', 1)
        # Each section starts with "<count> items" / "<count> elements", optionally followed by ':'
        profit = np.array(profit_part.split()[2:], dtype=int)
        weight = np.array(weight_part.split()[2:], dtype=int)
        relation = np.array(matrix_part.split(), dtype=int)
        if profit.size != m or weight.size != n or relation.size != m * n:
            raise ValueError(f"{path}: expected {m} profits, {n} weights, {m * n} matrix entries, "
                             f"got {profit.size}, {weight.size}, {relation.size}")
        name = os.path.basename(path)
        return cls(name, profit, weight, relation.reshape(m, n), capacity)

    def union_weight(self, y):
        """W(A_Y): weight of the union of the elements of the selected items (bool mask y)."""
        return self.weight[self.relation[y].any(axis=0)].sum()

    def evaluate(self, y):
        """:return: (profit, weight) of the item selection y (bool mask), feasible or not."""
        return self.profit[y].sum(), self.union_weight(y)


class Cover:
    """
    Items chosen so far and the union of their elements, updated incrementally.

    gain[i] is the weight item i would add if chosen now (its elements not yet
    covered). Gains only decrease, and since every element gets covered at most
    once, all the updates of one construction cost O(m * n) in total.
    """

    def __init__(self, inst):
        self.inst = inst
        self.selected = np.zeros(inst.m, dtype=bool)
        self.covered = np.zeros(inst.n, dtype=bool)
        self.load = 0.0
        self.profit = 0.0
        self.gain = inst.item_weight.copy()

    @classmethod
    def from_items(cls, inst, y):
        """Cover holding exactly the items of the bool mask y (feasibility is not checked)."""
        cover = cls(inst)
        cover.selected = y.copy()
        cover.covered = inst.relation[y].any(axis=0)
        cover.load = inst.weight[cover.covered].sum()
        cover.profit = inst.profit[y].sum()
        cover.gain = ((inst.weight32 * ~cover.covered) @ inst.relation_t).astype(float)
        return cover

    def fits(self, i):
        return self.load + self.gain[i] <= self.inst.capacity

    def add(self, i):
        """Selects item i (the caller checks fits(i))."""
        inst = self.inst
        elems = inst.subsets[i]
        new = elems[~self.covered[elems]]
        self.selected[i] = True
        self.profit += inst.profit[i]
        if new.size:
            self.covered[new] = True
            self.load += inst.weight[new].sum()
            self.gain -= inst.weight32[new] @ inst.relation_t[new]
