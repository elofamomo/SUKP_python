#include "instance.hpp"
#include "solver.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <instance.txt> [time_budget_sec=30] [rnum=2] [anum=2] [seed=1] [--print-solution]\n";
        return 1;
    }
    bool print_solution = false;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--print-solution") == 0) print_solution = true;
        else positional.push_back(argv[i]);
    }

    std::string path = positional[0];
    double time_budget = positional.size() > 1 ? std::stod(positional[1]) : 30.0;
    int rnum = positional.size() > 2 ? std::stoi(positional[2]) : 2;
    int anum = positional.size() > 3 ? std::stoi(positional[3]) : 2;
    uint64_t seed = positional.size() > 4 ? (uint64_t)std::stoull(positional[4]) : 1;

    Instance inst;
    try {
        inst = load_instance(path);
    } catch (const std::exception& e) {
        std::cerr << "failed to load instance: " << e.what() << "\n";
        return 1;
    }

    E2LSParams params;
    params.time_budget_sec = time_budget;
    params.rnum = rnum;
    params.anum = anum;
    params.seed = seed;

    E2LSSolver solver(inst, params);
    E2LSResult result = solver.run();

    long long weight = 0;
    {
        std::vector<int32_t> counts(inst.n, 0);
        for (int i = 0; i < inst.m; ++i) {
            if (!result.best_selected[i]) continue;
            for (int k = inst.subset_offset[i]; k < inst.subset_offset[i + 1]; ++k) {
                int e = inst.subset_flat[k];
                if (counts[e]++ == 0) weight += inst.weight[e];
            }
        }
    }

    std::cout << "instance=" << path << "\n";
    std::cout << "m=" << inst.m << " n=" << inst.n << " capacity=" << inst.capacity << "\n";
    std::cout << "profit=" << result.best_profit << "\n";
    std::cout << "weight=" << weight << "\n";
    std::cout << "feasible=" << (weight <= inst.capacity ? "true" : "false") << "\n";
    std::cout << "iterations=" << result.iterations << " restarts=" << result.restarts << "\n";
    std::cout << "elapsed_sec=" << std::fixed << std::setprecision(2) << result.elapsed_sec << "\n";
    if (print_solution) {
        std::cout << "solution=";
        for (int i = 0; i < inst.m; ++i) std::cout << (int)result.best_selected[i] << (i + 1 < inst.m ? " " : "\n");
    }
    return 0;
}
