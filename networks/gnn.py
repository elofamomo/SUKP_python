import torch
import torch.nn as nn


def mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


def aggregate(adj, h):
    """
    Batched message aggregation without copying adj per batch element.

    :param adj: torch.Tensor (a, b), weighted adjacency from b-nodes to a-nodes
    :param h: torch.Tensor (B, b, d), b-node embeddings
    :return: torch.Tensor (B, a, d)
    """
    B, b, d = h.shape
    out = adj @ h.transpose(0, 1).reshape(b, B * d)
    return out.reshape(-1, B, d).transpose(0, 1)


class MessagePassingLayer(nn.Module):
    """One round: items -> elements, elements -> items, then a global readout."""
    def __init__(self, hidden):
        super().__init__()
        self.elem_update = mlp(3 * hidden, hidden, hidden)
        self.item_update = mlp(3 * hidden, hidden, hidden)
        self.glob_update = mlp(3 * hidden, hidden, hidden)
        self.elem_norm = nn.LayerNorm(hidden)
        self.item_norm = nn.LayerNorm(hidden)
        self.glob_norm = nn.LayerNorm(hidden)

    def forward(self, h_item, h_elem, g, x, count, item_mean_adj, elem_mean_adj, elem_item_adj):
        # Elements read the mean of all their items, and separately the mean of their selected items
        agg_all = aggregate(elem_mean_adj, h_item)
        agg_sel = aggregate(elem_item_adj, h_item * x.unsqueeze(-1)) / count.clamp(min=1).unsqueeze(-1)
        h_elem = self.elem_norm(h_elem + self.elem_update(torch.cat([h_elem, agg_all, agg_sel], dim=-1)))

        agg_elem = aggregate(item_mean_adj, h_elem)
        g_item = g.unsqueeze(1).expand(-1, h_item.shape[1], -1)
        h_item = self.item_norm(h_item + self.item_update(torch.cat([h_item, agg_elem, g_item], dim=-1)))

        g = self.glob_norm(g + self.glob_update(torch.cat([g, h_item.mean(dim=1), h_elem.mean(dim=1)], dim=-1)))
        return h_item, h_elem, g


class BipartiteQNetwork(nn.Module):
    """
    Q-network over the SUKP item-element bipartite graph.

    Input: observation (m + 1) or (B, m + 1) = solution bits + best-so-far gap
    (see SetUnionHandler.get_observation).
    Output: Q-values (2m + 1) or (B, 2m + 1) = [add item i | remove item i | terminate].

    Node features are recomputed from the solution bits on every forward pass (coverage counts,
    marginal weights), then message passing runs over the fixed item-element edges. The add/remove
    heads are shared across items, so the parameter count does not depend on m or n.
    """
    ITEM_FEATURES = 6
    ELEM_FEATURES = 4
    GLOB_FEATURES = 4

    def __init__(self, item_profits, element_weights, item_subsets, capacity, hidden=64, layers=3):
        super().__init__()
        self.m = len(item_profits)
        self.n = len(element_weights)
        self.capacity = float(capacity)

        R = torch.zeros(self.m, self.n)
        for i, subset in enumerate(item_subsets):
            R[i, subset] = 1.0
        profits = torch.tensor(item_profits, dtype=torch.float32)
        weights = torch.tensor(element_weights, dtype=torch.float32)
        item_degree = R.sum(dim=1).clamp(min=1)
        elem_degree = R.sum(dim=0).clamp(min=1)

        # Instance data, not learned: kept out of checkpoints so a model can be loaded on another instance
        self.register_buffer('R', R, persistent=False)
        self.register_buffer('item_mean_adj', R / item_degree.unsqueeze(1), persistent=False)
        self.register_buffer('elem_mean_adj', R.t() / elem_degree.unsqueeze(1), persistent=False)
        self.register_buffer('elem_item_adj', R.t().contiguous(), persistent=False)
        self.register_buffer('elem_degree', elem_degree, persistent=False)
        self.register_buffer('weights', weights, persistent=False)
        self.register_buffer('profit_norm', profits / profits.mean(), persistent=False)
        self.register_buffer('weight_norm', weights / weights.mean(), persistent=False)
        # Weight unit: mean total weight of an item's elements
        self.weight_unit = float((R @ weights).mean())

        self.item_embed = mlp(self.ITEM_FEATURES, hidden, hidden)
        self.elem_embed = mlp(self.ELEM_FEATURES, hidden, hidden)
        self.glob_embed = mlp(self.GLOB_FEATURES, hidden, hidden)
        self.layers = nn.ModuleList([MessagePassingLayer(hidden) for _ in range(layers)])
        self.add_head = mlp(2 * hidden, hidden, 1)
        self.remove_head = mlp(2 * hidden, hidden, 1)
        self.terminate_head = mlp(hidden, hidden, 1)

    def node_features(self, obs):
        """
        :param obs: torch.Tensor (B, m + 1)
        :return: item (B, m, 6), element (B, n, 4), global (B, 4) features, plus x (B, m) and count (B, n)
        """
        x = obs[:, :self.m]
        gap = obs[:, self.m]

        count = x @ self.R                                   # selected items covering each element
        covered = (count > 0).float()
        unique = (count == 1).float()
        total_weight = covered @ self.weights
        residual = self.capacity - total_weight

        add_weight = ((1 - covered) * self.weights) @ self.R.t() * (1 - x)   # new weight if item is added
        free_weight = (unique * self.weights) @ self.R.t() * x                # weight freed if item is removed
        fits = ((add_weight <= residual.unsqueeze(1)) & (x == 0)).float()
        move_weight = add_weight + free_weight
        # Profit gained (add) or lost (remove) per unit of weight the move changes
        efficiency = torch.log(self.profit_norm) - torch.log(move_weight / self.weight_unit + 0.1)

        item = torch.stack([
            x,
            self.profit_norm.expand_as(x),
            add_weight / self.weight_unit,
            free_weight / self.weight_unit,
            fits,
            efficiency,
        ], dim=-1)
        elem = torch.stack([
            self.weight_norm.expand_as(count),
            covered,
            unique,
            count / self.elem_degree,
        ], dim=-1)
        glob = torch.stack([
            total_weight / self.capacity,
            residual / self.weight_unit,
            gap,
            x.mean(dim=1),
        ], dim=-1)
        return item, elem, glob, x, count

    def forward(self, obs):
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)

        item, elem, glob, x, count = self.node_features(obs)
        h_item = self.item_embed(item)
        h_elem = self.elem_embed(elem)
        g = self.glob_embed(glob)
        for layer in self.layers:
            h_item, h_elem, g = layer(h_item, h_elem, g, x, count,
                                      self.item_mean_adj, self.elem_mean_adj, self.elem_item_adj)

        g_item = g.unsqueeze(1).expand(-1, self.m, -1)
        head_in = torch.cat([h_item, g_item], dim=-1)
        q = torch.cat([
            self.add_head(head_in).squeeze(-1),
            self.remove_head(head_in).squeeze(-1),
            self.terminate_head(g),
        ], dim=-1)
        return q.squeeze(0) if single else q
