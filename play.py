"""
Run the C++ solver (cpp_solver/, through the sukp_cpp module) on the instance set in helper/config.yaml.

ADD* ranks add-back candidates either by the paper's value-weight ratio (--ranker ratio) or by the
trained ranker checkpoint (--ranker network, trained with train_ranker.py).

Build the module first (from the project root, inside the venv):
    cmake -S . -B build -DPython_EXECUTABLE=$(which python) && cmake --build build -j
"""
import argparse
import os
import numpy as np
import sukp_cpp
from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler

DEFAULT_CHECKPOINT = 'checkpoints/addback_ranker.pt'


def play(yaml_path="helper/config.yaml", ranker="ratio", checkpoint_path=DEFAULT_CHECKPOINT,
         time_budget=30.0, max_iterations=-1, rnum=2, anum=2, seed=1, device=None):
    """
    Solves the configured instance with the C++ local search.

    :param ranker: str, 'ratio' (paper ADD*) or 'network' (ADD* ranked by the trained ranker)
    :param checkpoint_path: str, ranker checkpoint, used when ranker == 'network'
    :param time_budget: float, seconds; <= 0 for no time limit
    :param max_iterations: int, local search steps; -1 for no limit
    :return: tuple (best_solution: np.array, best_profit: float, total_weight: float)
    """
    loader = SUKPLoader(yaml_path)
    inst = sukp_cpp.load_instance(loader.get_filepath())

    scorer = None
    if ranker == 'network':
        import torch
        from networks.ranker import InstanceGraph, NetworkScorer, load_ranker
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No trained ranker at {checkpoint_path}; train one with train_ranker.py")
        device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        model = load_ranker(checkpoint_path, device)
        scorer = NetworkScorer(model, InstanceGraph.from_instance(inst, device), device)
        print(f"Ranking add-back candidates with {checkpoint_path} on {device}")
    elif ranker != 'ratio':
        raise ValueError(f"ranker must be 'ratio' or 'network', got {ranker}")

    params = sukp_cpp.SolverParams()
    params.time_budget_sec = time_budget
    params.max_iterations = max_iterations
    params.rnum = rnum
    params.anum = anum
    params.seed = seed
    result = sukp_cpp.solve(inst, params, scorer)

    # Re-check the solution with the Python implementation; add_item refuses infeasible additions
    best_solution = result.best_selected.astype(float)
    suk = SetUnionHandler(loader.get_data(), loader.get_param())
    suk.set_state(best_solution)
    if suk.selected_items != set(np.flatnonzero(result.best_selected).tolist()) or suk.get_profit() != result.best_profit:
        raise RuntimeError("C++ solution does not match its Python re-evaluation")

    total_weight = suk.get_weight()
    print(f"Ranker: {ranker}, iterations: {result.iterations}, restarts: {result.restarts}, "
          f"add-back nodes: {result.add_back_nodes}, scorer calls: {result.scorer_calls}, "
          f"time: {result.elapsed_sec:.2f}s")
    print(f"Best Profit: {result.best_profit}")
    print(f"Total Weight: {total_weight:.0f} (Capacity: {suk.capacity:.0f})")
    print(f"Feasible: {suk.is_feasible()}")
    print(f"Solution: {' '.join(['1' if x > 0.5 else '0' for x in best_solution])}")
    return best_solution, result.best_profit, total_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the instance in --config with the C++ local search")
    parser.add_argument('--config', default='helper/config.yaml')
    parser.add_argument('--ranker', choices=['ratio', 'network'], default='ratio')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--time', type=float, default=30.0, help='seconds; <= 0 for no limit')
    parser.add_argument('--iterations', type=int, default=-1, help='local search steps; -1 for no limit')
    parser.add_argument('--rnum', type=int, default=2)
    parser.add_argument('--anum', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    play(args.config, args.ranker, args.checkpoint, args.time, args.iterations,
         args.rnum, args.anum, args.seed, args.device)
