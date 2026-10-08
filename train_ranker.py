"""
Train the add-back ranker (networks/ranker.py) by imitating exhaustive add-back search.

1. Collect: run the C++ solver (paper ranking) on training instances. At random removal trials it
   labels each add-back decision on the exact-optimal path with the best profit reachable after
   adding each candidate (see SolverParams.sample_prob and oracle_node_budget in cpp_solver/solver.hpp).
2. Train: listwise cross-entropy between the ranker's candidate scores and the optimal candidates,
   validated on held-out instances.
3. Save the best validation checkpoint (default checkpoints/addback_ranker.pt) for play.py --ranker network.

By default it trains on generated instances (helper/instance_generator.py), so the 60 benchmark
instances stay unseen; pass --instances to train on instance files instead. With --epochs 0 it only
reports how often the paper's ratio ranking already picks an optimal candidate.

Example:
    python train_ranker.py --samples data/addback_samples.pt --epochs 30
"""
import argparse
import glob
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from networks.ranker import AddBackRanker, InstanceGraph, save_ranker


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = p.add_argument_group('training instances')
    data.add_argument('--instances', nargs='+', help='instance file globs (default: generate instances)')
    data.add_argument('--sizes', default='85x100,100x85,100x100,185x200,200x185,200x200',
                      help='generated sizes, comma-separated MxN')
    data.add_argument('--alpha-beta', default='0.10:0.75,0.15:0.85',
                      help='generated (density alpha, capacity ratio beta) pairs, comma-separated')
    data.add_argument('--per-setting', type=int, default=4, help='generated instances per size and alpha-beta pair')

    collect = p.add_argument_group('label collection')
    collect.add_argument('--samples', help='cache of collected samples: loaded if it exists, else written after collecting')
    collect.add_argument('--collect-iterations', type=int, default=2000, help='local search steps per instance')
    collect.add_argument('--sample-prob', type=float, default=0.05, help='chance of labelling each removal trial')
    collect.add_argument('--max-samples', type=int, default=300, help='samples per instance')
    collect.add_argument('--oracle-budget', type=int, default=200000, help='exhaustive search nodes per label')
    collect.add_argument('--rnum', type=int, default=2)
    collect.add_argument('--anum', type=int, default=2, help='also k for the hit@k metric')

    train = p.add_argument_group('training')
    train.add_argument('--epochs', type=int, default=30)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--lr', type=float, default=1e-3)
    train.add_argument('--hidden', type=int, default=64)
    train.add_argument('--layers', type=int, default=3)
    train.add_argument('--val-frac', type=float, default=0.2, help='fraction of instances held out')
    train.add_argument('--out', default='checkpoints/addback_ranker.pt')
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def instance_spec(name, profits, weights, subsets, capacity):
    return {'name': name, 'profits': np.asarray(profits), 'weights': np.asarray(weights),
            'subsets': [np.asarray(s) for s in subsets], 'capacity': int(capacity)}


def build_instances(args, rng):
    """:return: list of (instance spec, sukp_cpp.Instance)"""
    import sukp_cpp
    if args.instances:
        paths = sorted({path for pattern in args.instances for path in glob.glob(pattern)})
        if not paths:
            raise ValueError(f"No instance files match {args.instances}")
        out = []
        for path in paths:
            inst = sukp_cpp.load_instance(path)
            name = os.path.splitext(os.path.basename(path))[0]
            out.append((instance_spec(name, inst.profits, inst.weights, inst.subsets, inst.capacity), inst))
        return out

    from helper.instance_generator import generate_instance
    out = []
    for size in args.sizes.split(','):
        m, n = (int(v) for v in size.split('x'))
        for pair in args.alpha_beta.split(','):
            alpha, beta = (float(v) for v in pair.split(':'))
            for k in range(args.per_setting):
                profits, weights, subsets, capacity = generate_instance(m, n, alpha, beta, rng)
                inst = sukp_cpp.make_instance(profits.tolist(), weights.tolist(), [s.tolist() for s in subsets], capacity)
                name = f"gen_{m}_{n}_{alpha:.2f}_{beta:.2f}_{k}"
                out.append((instance_spec(name, profits, weights, subsets, capacity), inst))
    return out


def collect_samples(args, rng):
    """:return: dict(instances=list of specs, samples=list of (instance index, selected, candidates, values))"""
    import sukp_cpp
    instances, samples = [], []
    for idx, (spec, inst) in enumerate(build_instances(args, rng)):
        params = sukp_cpp.SolverParams()
        params.rnum, params.anum = args.rnum, args.anum
        params.time_budget_sec = 0
        params.max_iterations = args.collect_iterations
        params.seed = args.seed + idx
        params.sample_prob = args.sample_prob
        params.max_samples = args.max_samples
        params.oracle_node_budget = args.oracle_budget
        start = time.time()
        res = sukp_cpp.solve(inst, params)
        instances.append(spec)
        samples.extend((idx, selected, candidates, values) for selected, candidates, values in res.samples)
        print(f"[collect] {spec['name']}: {len(res.samples)} samples, best profit {res.best_profit}, "
              f"{time.time() - start:.1f}s")
    return {'instances': instances, 'samples': samples}


def load_or_collect(args, rng):
    if args.samples and os.path.exists(args.samples):
        print(f"Loading samples from {args.samples}")
        return torch.load(args.samples, weights_only=False)
    data = collect_samples(args, rng)
    if args.samples:
        os.makedirs(os.path.dirname(args.samples) or '.', exist_ok=True)
        torch.save(data, args.samples)
        print(f"Saved samples to {args.samples}")
    return data


def make_batches(samples, indices, batch_size, rng):
    """Batches never mix instances, since a forward pass runs on one instance graph."""
    by_instance = {}
    for i in indices:
        by_instance.setdefault(samples[i][0], []).append(i)
    batches = []
    for inst_idx, idxs in by_instance.items():
        idxs = list(rng.permutation(idxs))
        batches.extend((inst_idx, idxs[k:k + batch_size]) for k in range(0, len(idxs), batch_size))
    order = rng.permutation(len(batches))
    return [batches[k] for k in order]


def batch_stats(model, graph, batch, anum, device):
    """
    :param batch: list of (selected, candidates, values)
    :return: loss tensor, dict of summed metrics over the batch (model and paper-ratio rankings)
    """
    x = torch.as_tensor(np.stack([s for s, _, _ in batch]), dtype=torch.float32, device=device)
    scores = model(x, graph)

    K = max(len(c) for _, c, _ in batch)
    cand = torch.zeros(len(batch), K, dtype=torch.long, device=device)
    vals = torch.full((len(batch), K), -1, dtype=torch.long, device=device)
    for b, (_, c, v) in enumerate(batch):
        cand[b, :len(c)] = torch.as_tensor(c, dtype=torch.long)
        vals[b, :len(v)] = torch.as_tensor(v, dtype=torch.long)
    mask = vals >= 0

    logits = scores.gather(1, cand).masked_fill(~mask, float('-inf'))
    best = vals.max(dim=1, keepdim=True).values
    optimal = (vals == best) & mask
    target = optimal.float() / optimal.float().sum(dim=1, keepdim=True)
    log_probs = F.log_softmax(logits, dim=1).masked_fill(~mask, 0.0)
    loss = -(target * log_probs).sum(dim=1).mean()

    with torch.no_grad():
        k = min(anum, K)
        top = logits.topk(k, dim=1).indices
        top_ok = optimal.gather(1, top)
        chosen = vals.gather(1, top[:, :1]).squeeze(1).float()
        best_f = best.squeeze(1).float().clamp(min=1)
        # Candidates are stored in ratio order, so the paper's top-k is the first k columns
        stats = {
            'n': len(batch),
            'loss': loss.item() * len(batch),
            'top1': top_ok[:, 0].float().sum().item(),
            'hit': top_ok.any(dim=1).float().sum().item(),
            'regret': ((best_f - chosen) / best_f).sum().item(),
            'ratio_top1': optimal[:, 0].float().sum().item(),
            'ratio_hit': optimal[:, :k].any(dim=1).float().sum().item(),
            'ratio_regret': ((best_f - vals[:, 0].float()) / best_f).sum().item(),
        }
    return loss, stats


def evaluate(model, graphs, data, indices, args, rng):
    totals = {}
    model.eval()
    with torch.no_grad():
        for inst_idx, idxs in make_batches(data['samples'], indices, args.batch_size, rng):
            batch = [data['samples'][i][1:] for i in idxs]
            _, stats = batch_stats(model, graphs[inst_idx], batch, args.anum, args.device)
            for key, v in stats.items():
                totals[key] = totals.get(key, 0.0) + v
    n = max(totals.get('n', 0), 1)
    return {key: v / n for key, v in totals.items() if key != 'n'}


def format_stats(stats, anum):
    return (f"loss {stats['loss']:.4f} | network top1 {stats['top1']:.3f} hit@{anum} {stats['hit']:.3f} "
            f"regret {stats['regret']:.5f} | ratio top1 {stats['ratio_top1']:.3f} hit@{anum} "
            f"{stats['ratio_hit']:.3f} regret {stats['ratio_regret']:.5f}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    data = load_or_collect(args, np.random.default_rng(args.seed))
    # Fresh generator, so the split is the same whether samples were collected or loaded
    rng = np.random.default_rng(args.seed)
    instances, samples = data['instances'], data['samples']
    if not samples:
        raise ValueError("No samples collected; raise --sample-prob, --collect-iterations or --oracle-budget")

    inst_order = rng.permutation(len(instances))
    n_val = math.ceil(args.val_frac * len(instances)) if len(instances) > 1 else 0
    val_insts = set(inst_order[:n_val].tolist())
    train_idx = [i for i, s in enumerate(samples) if s[0] not in val_insts]
    val_idx = [i for i, s in enumerate(samples) if s[0] in val_insts]
    print(f"{len(instances)} instances ({len(val_insts)} held out), "
          f"{len(train_idx)} train / {len(val_idx)} val samples")

    graphs = {i: InstanceGraph(spec['profits'], spec['weights'], spec['subsets'], spec['capacity'], args.device)
              for i, spec in enumerate(instances)}
    model = AddBackRanker(hidden=args.hidden, layers=args.layers).to(args.device)
    eval_idx = val_idx or train_idx
    print(f"untrained ({'val' if val_idx else 'train'}): {format_stats(evaluate(model, graphs, data, eval_idx, args, rng), args.anum)}")
    if args.epochs == 0:
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {}
        for inst_idx, idxs in make_batches(samples, train_idx, args.batch_size, rng):
            batch = [samples[i][1:] for i in idxs]
            loss, stats = batch_stats(model, graphs[inst_idx], batch, args.anum, args.device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, v in stats.items():
                totals[key] = totals.get(key, 0.0) + v
        train_stats = {key: v / totals['n'] for key, v in totals.items() if key != 'n'}
        print(f"epoch {epoch} train: {format_stats(train_stats, args.anum)}")

        if val_idx:
            val_stats = evaluate(model, graphs, data, val_idx, args, rng)
            print(f"epoch {epoch} val:   {format_stats(val_stats, args.anum)}")
            if val_stats['loss'] < best_val:
                best_val = val_stats['loss']
                save_ranker(model, args.out, epoch=epoch, val_stats=val_stats, train_args=vars(args))
                print(f"Saved {args.out}")
        else:
            save_ranker(model, args.out, epoch=epoch, train_args=vars(args))


if __name__ == "__main__":
    main()
