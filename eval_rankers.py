"""
Compare ADD* rankers on the 60 benchmark instances: the paper's value-weight ratio ('ratio') and the
imitation-trained ranker ('network', train_ranker.py).

Each job runs the C++ local search once on one instance with one ranker and seed, under either a step
budget (--iterations) or a time budget (--time), and appends one row to --out. Rows already in --out are
skipped, so an interrupted run resumes where it stopped.

Example:
    python eval_rankers.py --iterations 1000 --seeds 1 2 3 --out results_csv/rankers_iter1000.csv
    python eval_rankers.py --time 30 --seeds 1 2 3 --out results_csv/rankers_time30.csv
"""
import argparse
import csv
import glob
import os
import time
from multiprocessing import Pool

FIELDS = ['set', 'instance', 'm', 'n', 'ranker', 'seed', 'budget', 'profit', 'iterations', 'restarts',
          'add_back_nodes', 'scorer_calls', 'elapsed_sec']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    budget = p.add_mutually_exclusive_group(required=True)
    budget.add_argument('--iterations', type=int, help='local search steps per run')
    budget.add_argument('--time', type=float, help='seconds per run')
    p.add_argument('--rankers', nargs='+', default=['ratio', 'network'], choices=['ratio', 'network'])
    p.add_argument('--seeds', nargs='+', type=int, default=[1])
    p.add_argument('--checkpoint', default='checkpoints/addback_ranker.pt')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', required=True)
    return p.parse_args()


def run(job):
    path, ranker, seed, iterations, time_budget, checkpoint = job
    import torch
    torch.set_num_threads(1)  # one core per worker
    import sukp_cpp
    inst = sukp_cpp.load_instance(path)
    scorer = None
    if ranker == 'network':
        from networks.ranker import InstanceGraph, NetworkScorer, load_ranker
        device = torch.device('cpu')
        scorer = NetworkScorer(load_ranker(checkpoint, device), InstanceGraph.from_instance(inst, device), device)

    params = sukp_cpp.SolverParams()
    params.time_budget_sec = time_budget if time_budget is not None else 0
    params.max_iterations = iterations if iterations is not None else -1
    params.seed = seed
    start = time.time()
    res = sukp_cpp.solve(inst, params, scorer)
    name = os.path.basename(path)
    _, m, n, _, _ = os.path.splitext(name)[0].split('_')
    return {
        'set': 'Set I' if 'Set I/' in path else 'Set II',
        'instance': name, 'm': m, 'n': n, 'ranker': ranker, 'seed': seed,
        'budget': f'{iterations} it' if iterations is not None else f'{time_budget:g} s',
        'profit': res.best_profit, 'iterations': res.iterations, 'restarts': res.restarts,
        'add_back_nodes': res.add_back_nodes, 'scorer_calls': res.scorer_calls,
        'elapsed_sec': f'{time.time() - start:.2f}',
    }


def main():
    args = parse_args()
    paths = sorted(glob.glob('SUKP_instances_60/Instances of Set I/*.txt')) + \
        sorted(glob.glob('SUKP_instances_60/Instances of Set II/*.txt'))
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r['instance'], r['ranker'], int(r['seed'])) for r in csv.DictReader(f)}
    jobs = [(path, ranker, seed, args.iterations, args.time, args.checkpoint)
            for seed in args.seeds for path in paths for ranker in args.rankers
            if (os.path.basename(path), ranker, seed) not in done]
    print(f"{len(jobs)} runs to do, {len(done)} already in {args.out}")

    new_file = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as f, Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for k, row in enumerate(pool.imap_unordered(run, jobs), 1):
            writer.writerow(row)
            f.flush()
            print(f"[{k}/{len(jobs)}] {row['instance']} {row['ranker']} seed {row['seed']}: "
                  f"profit {row['profit']}, {row['iterations']} it, {row['elapsed_sec']} s", flush=True)


if __name__ == "__main__":
    main()
