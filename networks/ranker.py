import numpy as np
import torch
import torch.nn as nn
from networks.gnn import mlp, MessagePassingLayer, BipartiteQNetwork


class InstanceGraph:
    """Dense item-element tensors of one instance, shared by every forward pass on it."""
    def __init__(self, profits, weights, subsets, capacity, device='cpu'):
        self.m = len(profits)
        self.n = len(weights)
        self.capacity = float(capacity)

        R = np.zeros((self.m, self.n), dtype=np.float32)
        for i, subset in enumerate(subsets):
            R[i, np.asarray(subset, dtype=np.int64)] = 1.0
        R = torch.from_numpy(R).to(device)
        profits = torch.as_tensor(np.asarray(profits, dtype=np.float32), device=device)
        weights = torch.as_tensor(np.asarray(weights, dtype=np.float32), device=device)
        item_degree = R.sum(dim=1).clamp(min=1)
        elem_degree = R.sum(dim=0).clamp(min=1)

        self.R = R
        self.item_mean_adj = R / item_degree.unsqueeze(1)
        self.elem_mean_adj = R.t() / elem_degree.unsqueeze(1)
        self.elem_item_adj = R.t().contiguous()
        self.elem_degree = elem_degree
        self.weights = weights
        self.profit_norm = profits / profits.mean()
        self.weight_norm = weights / weights.mean()
        # Weight unit: mean total weight of an item's elements
        self.weight_unit = float((R @ weights).mean())

    @classmethod
    def from_instance(cls, inst, device='cpu'):
        """:param inst: sukp_cpp.Instance"""
        return cls(inst.profits, inst.weights, inst.subsets, inst.capacity, device)


class AddBackRanker(nn.Module):
    """
    Scores add-back candidates for ADD* in cpp_solver/solver.hpp over the item-element bipartite graph.

    Input: selection bits x (m) or (B, m) at an add-back node, after its free items were added.
    Output: one score per item (m) or (B, m); only the feasible candidates' scores are used, higher first.

    The score is log(profit / added weight), i.e. the paper's value-weight ratio, plus a learned
    correction whose output layer starts at zero, so an untrained model ranks like the paper and
    training learns where to deviate (e.g. items whose elements overlap). No parameter depends on
    m or n, so one checkpoint serves every instance.
    """
    ITEM_FEATURES = 7
    ELEM_FEATURES = 4
    GLOB_FEATURES = 3

    def __init__(self, hidden=64, layers=3):
        super().__init__()
        self.config = {'hidden': hidden, 'layers': layers}
        self.item_embed = mlp(self.ITEM_FEATURES, hidden, hidden)
        self.elem_embed = mlp(self.ELEM_FEATURES, hidden, hidden)
        self.glob_embed = mlp(self.GLOB_FEATURES, hidden, hidden)
        self.layers = nn.ModuleList([MessagePassingLayer(hidden) for _ in range(layers)])
        self.score_head = mlp(2 * hidden, hidden, 1)
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def node_features(self, x, graph):
        """
        :param x: torch.Tensor (B, m)
        :return: item (B, m, 7), element (B, n, 4), global (B, 3) features, count (B, n), log value-weight ratio (B, m)
        """
        count = x @ graph.R                                  # selected items covering each element
        covered = (count > 0).float()
        unique = (count == 1).float()
        total_weight = covered @ graph.weights
        residual = graph.capacity - total_weight

        add_weight = ((1 - covered) * graph.weights) @ graph.R.t() * (1 - x)   # new weight if item is added
        free_weight = (unique * graph.weights) @ graph.R.t() * x                # weight freed if item is removed
        fits = ((add_weight <= residual.unsqueeze(1)) & (x == 0)).float()
        candidate = fits * (add_weight > 0).float()          # the C++ feasible candidate set FC
        efficiency = torch.log(graph.profit_norm) - torch.log((add_weight + free_weight) / graph.weight_unit + 0.1)
        # Undefined for non-candidates (no added weight); those scores are never used
        log_ratio = torch.log(graph.profit_norm) - torch.log((add_weight / graph.weight_unit).clamp(min=1e-6))

        item = torch.stack([
            x,
            graph.profit_norm.expand_as(x),
            add_weight / graph.weight_unit,
            free_weight / graph.weight_unit,
            fits,
            candidate,
            efficiency,
        ], dim=-1)
        elem = torch.stack([
            graph.weight_norm.expand_as(count),
            covered,
            unique,
            count / graph.elem_degree,
        ], dim=-1)
        glob = torch.stack([
            total_weight / graph.capacity,
            residual / graph.weight_unit,
            x.mean(dim=1),
        ], dim=-1)
        return item, elem, glob, count, log_ratio

    def forward(self, x, graph):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        item, elem, glob, count, log_ratio = self.node_features(x, graph)
        h_item = self.item_embed(item)
        h_elem = self.elem_embed(elem)
        g = self.glob_embed(glob)
        for layer in self.layers:
            h_item, h_elem, g = layer(h_item, h_elem, g, x, count,
                                      graph.item_mean_adj, graph.elem_mean_adj, graph.elem_item_adj)

        g_item = g.unsqueeze(1).expand(-1, graph.m, -1)
        scores = log_ratio + self.score_head(torch.cat([h_item, g_item], dim=-1)).squeeze(-1)
        return scores.squeeze(0) if single else scores


class NetworkScorer:
    """The scorer passed to sukp_cpp.solve: scores the candidates of one add-back node."""
    def __init__(self, model, graph, device):
        self.model = model.eval()
        self.graph = graph
        self.device = device

    @torch.no_grad()
    def __call__(self, selected, candidates):
        """
        :param selected: np.array[uint8] (m), current selection
        :param candidates: np.array[int32] (k), feasible candidates
        :return: np.array[float64] (k), higher is branched on first
        """
        x = torch.as_tensor(selected, dtype=torch.float32, device=self.device)
        scores = self.model(x, self.graph)
        idx = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
        return scores[idx].double().cpu().numpy()


class DQNScorer:
    """
    The scorer passed to sukp_cpp.solve for --ranker dqn: ranks add-back candidates by Q(s, add i)
    of a DQN pretrained on the instance (BipartiteQNetwork, trained by train.py).
    """
    def __init__(self, model, device):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, selected, candidates):
        # Observation = solution bits + gap to the episode's best (SetUnionHandler.get_observation).
        # ADD* only adds items, so every node is scored as if it were at the best so far (gap 0).
        obs = torch.zeros(self.model.m + 1, device=self.device)
        obs[:self.model.m] = torch.as_tensor(selected, dtype=torch.float32, device=self.device)
        q = self.model(obs)  # [add item i | remove item i | terminate]
        idx = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
        return q[idx].double().cpu().numpy()


def load_dqn(path, data, device):
    """
    :param path: str, checkpoint saved by train.py
    :param data: dict from SUKPLoader.get_data() for the instance the DQN was trained on
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = BipartiteQNetwork(data['item_profits'], data['element_weights'],
                              data['item_subsets'], data['capacity']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval()


def save_ranker(model, path, **extra):
    torch.save({'model_state_dict': model.state_dict(), 'model_config': model.config, **extra}, path)


def load_ranker(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = AddBackRanker(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval()
