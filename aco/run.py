"""
Runs BABC or MMAS on the benchmark instances and compares with the literature.

Usage (from the repo root):
    python -m aco.run --algo babc --set 1 --runs 100
    python -m aco.run --algo mmas --set 1 --runs 10 --instances sukp_100_85_0.10_0.75 --beta 3
    python -m aco.run --algo mmas --backend py ...   # the Python reference implementation
Writes aco/results/<algo>_set<k>.csv (per instance: Best, Mean, StD, Time over the
runs, as in Tables 2-4 of He et al. 2018) and prints it next to the published BABC
results (aco/babc_paper.csv, Set I only) and the best known value (results_csv/profit.csv).

The C++ backend (aco/aco_cpp*.so) is built with the rest of the project:
    cmake -S . -B build -DPython_EXECUTABLE=$(which python) && cmake --build build -j
"""
import os
# One BLAS thread per process: the matrix-vector products are small, the runs go in parallel
for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(var, '1')

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from aco.instance import Instance
from aco.babc import babc
from aco.mmas import mmas

SETS = {1: 'SUKP_instances_60/Instances of Set I', 2: 'SUKP_instances_60/Instances of Set II'}
LIT = ['babc', 'i2pls', 'kbts', 'e2ls', 'ql', 'gbpso', 'aes_bm']


def run_once(path, algo, backend, params, seed):
    inst = Instance.from_file(path)
    if backend == 'cpp':
        from aco import aco_cpp
        res = getattr(aco_cpp, algo)(path, seed=seed, **params)
        elapsed = res['elapsed']
    else:
        t0 = time.perf_counter()
        res = {'babc': babc, 'mmas': mmas}[algo](inst, rng=np.random.default_rng(seed), **params)
        elapsed = time.perf_counter() - t0
    # Checked independently of the backend that produced the solution
    profit, weight = inst.evaluate(res['solution'])
    if weight > inst.capacity or profit != res['profit']:
        raise RuntimeError(f"{inst.name}: invalid solution (profit {profit} vs {res['profit']}, "
                           f"weight {weight} > {inst.capacity})")
    return inst.name, profit, elapsed, res['evaluations']


def load_reference():
    """Published BABC (Best, Mean) and the best known value per instance, when available."""
    paper, bks = {}, {}
    with open(os.path.join(os.path.dirname(__file__), 'babc_paper.csv')) as f:
        for r in csv.DictReader(f):
            paper[r['instance']] = (float(r['best']), float(r['mean']))
    if os.path.exists('results_csv/profit.csv'):
        with open('results_csv/profit.csv') as f:
            for r in csv.DictReader(f):
                vals = [int(r[k]) for k in LIT if r[k]]
                if vals:
                    bks[r['instance']] = max(vals)
    return paper, bks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--algo', choices=['babc', 'mmas'], required=True)
    ap.add_argument('--backend', choices=['cpp', 'py'], default='cpp')
    ap.add_argument('--set', type=int, choices=[1, 2], default=1)
    ap.add_argument('--instances', nargs='*', help='instance names (without .txt); default: the whole set')
    ap.add_argument('--runs', type=int, default=100, help='independent runs per instance (paper: 100)')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() // 2),
                    help='parallel processes (default: half the CPUs, leaving room for the desktop)')
    ap.add_argument('--seed', type=int, default=0, help='run r uses seed + r')
    ap.add_argument('--max-iter', type=int, help='default max(m, n)')
    ap.add_argument('--pop', type=int, default=20, help='bees / ants per iteration (paper: 20)')
    ap.add_argument('--alpha', type=float, default=1.0, help='MMAS pheromone exponent')
    ap.add_argument('--beta', type=float, default=2.0, help='MMAS heuristic exponent')
    ap.add_argument('--rho', type=float, default=0.1, help='MMAS evaporation rate')
    ap.add_argument('--tau-min', type=float, default=0.01, help='MMAS lower pheromone bound')
    ap.add_argument('--heuristic', choices=['static', 'dynamic'], default='static', help='MMAS eta')
    ap.add_argument('--out', help='default aco/results/<algo>_set<k>.csv')
    args = ap.parse_args()

    if args.algo == 'babc':
        params = {'n_bees': args.pop, 'max_iter': args.max_iter}
    else:
        params = {'n_ants': args.pop, 'max_iter': args.max_iter, 'alpha': args.alpha, 'beta': args.beta,
                  'rho': args.rho, 'tau_min': args.tau_min, 'heuristic': args.heuristic}

    folder = SETS[args.set]
    names = [n + '.txt' for n in args.instances] if args.instances else sorted(os.listdir(folder))
    paths = {name: os.path.join(folder, name) for name in names}
    sizes = {name: os.path.getsize(p) for name, p in paths.items()}
    # Largest instances first, so the slow runs do not end up alone at the tail
    jobs = [(paths[name], args.algo, args.backend, params, args.seed + r)
            for name in sorted(names, key=sizes.get, reverse=True) for r in range(args.runs)]

    results = {name: [] for name in names}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_once, *job) for job in jobs]
        for done, fut in enumerate(futures, 1):
            name, profit, elapsed, evals = fut.result()
            results[name].append((profit, elapsed, evals))
            if len(results[name]) == args.runs:
                print(f"[{done}/{len(jobs)}] {name} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    paper, bks = load_reference()
    rows = []
    for name in sorted(names, key=lambda s: tuple(float(x) for x in s[5:-4].split('_'))):
        profits, times, evals = (np.array(col) for col in zip(*results[name]))
        p_best, p_mean = paper.get(name, (None, None))
        rows.append({
            'instance': name[:-4], 'best': int(profits.max()), 'mean': round(profits.mean(), 2),
            'std': round(profits.std(), 2), 'time': round(times.mean(), 3), 'evals': int(evals.mean()),
            'babc_paper_best': p_best, 'babc_paper_mean': p_mean, 'bks': bks.get(name),
            'gap_best_%': round(100 * (bks[name] - profits.max()) / bks[name], 2) if name in bks else None,
        })

    out = args.out or f"aco/results/{args.algo}_set{args.set}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{args.algo} ({args.backend}) {params}, {args.runs} runs")
    print(f"{'instance':<24}{'best':>7}{'mean':>10}{'std':>8}{'time':>8} | {'BABC best':>9}{'BABC mean':>10} | {'BKS':>6}{'gap%':>6}")
    for r in rows:
        pb = f"{r['babc_paper_best']:.0f}" if r['babc_paper_best'] else '-'
        pm = f"{r['babc_paper_mean']:.1f}" if r['babc_paper_mean'] else '-'
        print(f"{r['instance']:<24}{r['best']:>7}{r['mean']:>10.1f}{r['std']:>8.1f}{r['time']:>8.2f} | "
              f"{pb:>9}{pm:>10} | {r['bks'] or '-':>6}{r['gap_best_%'] if r['gap_best_%'] is not None else '-':>6}")
    print(f"saved to {out}")


if __name__ == '__main__':
    main()
