// mmas_run: runs MMAS (mmas.hpp) on the SUKP benchmark instances, many runs in parallel threads,
// and compares with the best known values.
//
// Build and run from the repo root:
//   cmake -S mmas -B mmas/build && cmake --build mmas/build -j
//   ./mmas/build/mmas_run --set 1 --runs 100
//   ./mmas/build/mmas_run --set 2 --runs 100 --heuristic dynamic
//   ./mmas/build/mmas_run --instances sukp_100_85_0.10_0.75 sukp_500_500_0.15_0.85 --runs 10
// Writes mmas/results/mmas_set<k>.csv (per instance: Best, Mean, StD, Time over the runs and the
// gaps to the best known value) and mmas/results/mmas_set<k>_runs.csv (one row per run).
#include "instance.hpp"
#include "mmas.hpp"
#include <atomic>
#include <thread>
#include <mutex>
#include <filesystem>
#include <iostream>
#include <map>
#include <cstdio>
#include <cstring>

namespace fs = std::filesystem;

namespace {

const char* USAGE = R"(usage: mmas_run [options]
  --set 1|2              benchmark set (default 1)
  --instances NAME...    instance names without .txt, from either set (default: the whole --set)
  --runs N               independent runs per instance (default 100)
  --threads T            parallel runs (default: half the CPUs)
  --seed S               run r uses seed S + r (default 0)
  --out PATH             summary CSV (default mmas/results/mmas_set<k>.csv); per-run rows go to <PATH>_runs.csv
  --overwrite            replace --out if it exists (default: refuse, to protect saved results)
  --ants N               ants per iteration (default 20)
  --max-iter N           iterations (default max(m, n))
  --iter-factor F        iterations = F * max(m, n), e.g. 0.4 with --ants 50 keeps the default budget
  --alpha A              pheromone exponent (default 1)
  --beta B               heuristic exponent (default 2)
  --rho R                evaporation rate (default 0.1)
  --tau-min T            lower pheromone bound, the upper is 1 (default 0.01)
  --heuristic static|dynamic   eta = p/R, or p / weight the item adds now (default static)
  --gb-every K           every K-th iteration the global best deposits pheromone (default 0: never)
  --restart K            reset pheromone after K iterations without improvement (default 0: never)
)";

struct Options {
    int set = 1;
    std::vector<std::string> instances;
    int runs = 100;
    int threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    uint64_t seed = 0;
    std::string out;
    bool overwrite = false;
    double iter_factor = 0;  // > 0: max_iter = iter_factor * max(m, n), per instance
    aco::MMASParams mmas;
};

Options parse_args(int argc, char** argv) {
    Options o;
    for (int k = 1; k < argc; ++k) {
        std::string a = argv[k];
        auto value = [&]() -> std::string {
            if (k + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++k];
        };
        if (a == "--help" || a == "-h") { std::cout << USAGE; std::exit(0); }
        else if (a == "--set") o.set = std::stoi(value());
        else if (a == "--instances") { while (k + 1 < argc && std::strncmp(argv[k + 1], "--", 2) != 0) o.instances.push_back(argv[++k]); }
        else if (a == "--runs") o.runs = std::stoi(value());
        else if (a == "--threads") o.threads = std::stoi(value());
        else if (a == "--seed") o.seed = std::stoull(value());
        else if (a == "--out") o.out = value();
        else if (a == "--overwrite") o.overwrite = true;
        else if (a == "--ants") o.mmas.n_ants = std::stoi(value());
        else if (a == "--max-iter") o.mmas.max_iter = std::stoi(value());
        else if (a == "--iter-factor") o.iter_factor = std::stod(value());
        else if (a == "--alpha") o.mmas.alpha = std::stod(value());
        else if (a == "--beta") o.mmas.beta = std::stod(value());
        else if (a == "--rho") o.mmas.rho = std::stod(value());
        else if (a == "--tau-min") o.mmas.tau_min = std::stod(value());
        else if (a == "--heuristic") {
            std::string h = value();
            if (h != "static" && h != "dynamic") throw std::invalid_argument("--heuristic must be static or dynamic");
            o.mmas.dynamic = h == "dynamic";
        }
        else if (a == "--gb-every") o.mmas.gb_every = std::stoi(value());
        else if (a == "--restart") o.mmas.restart = std::stoi(value());
        else throw std::invalid_argument("unknown option " + a);
    }
    if (o.set != 1 && o.set != 2) throw std::invalid_argument("--set must be 1 or 2");
    if (o.runs < 1 || o.threads < 1 || o.mmas.n_ants < 1)
        throw std::invalid_argument("--runs, --threads and --ants must be positive");
    if (o.iter_factor > 0 && o.mmas.max_iter > 0)
        throw std::invalid_argument("give --max-iter or --iter-factor, not both");
    if (o.out.empty()) o.out = "mmas/results/mmas_set" + std::to_string(o.set) + ".csv";
    if (o.out.size() < 5 || o.out.substr(o.out.size() - 4) != ".csv") throw std::invalid_argument("--out must end in .csv");
    return o;
}

// Profit and union weight of a selection, recomputed from scratch (independent of aco::Cover)
std::pair<long long, long long> evaluate(const Instance& inst, const std::vector<uint8_t>& sel) {
    std::vector<uint8_t> covered(inst.n, 0);
    long long profit = 0, weight = 0;
    for (int i = 0; i < inst.m; ++i) {
        if (!sel[i]) continue;
        profit += inst.profit[i];
        for (int k = inst.subset_offset[i]; k < inst.subset_offset[i + 1]; ++k) {
            int e = inst.subset_flat[k];
            if (!covered[e]) { covered[e] = 1; weight += inst.weight[e]; }
        }
    }
    return {profit, weight};
}

// Comma-separated rows with a header (the reference files use no quoting)
std::vector<std::map<std::string, std::string>> read_csv(const std::string& path) {
    std::vector<std::map<std::string, std::string>> rows;
    std::ifstream f(path);
    if (!f) return rows;
    auto split = [](const std::string& line) {
        std::vector<std::string> out;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) out.push_back(cell);
        if (!line.empty() && line.back() == ',') out.push_back("");
        return out;
    };
    std::string line;
    std::getline(f, line);
    std::vector<std::string> header = split(line);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cells = split(line);
        std::map<std::string, std::string> row;
        for (size_t c = 0; c < header.size(); ++c) row[header[c]] = c < cells.size() ? cells[c] : "";
        rows.push_back(row);
    }
    return rows;
}

// sukp_100_85_0.10_0.75 -> (100, 85, 0.10, 0.75), the order of the result tables
std::vector<double> name_key(const std::string& name) {
    std::vector<double> key;
    std::stringstream ss(name.substr(5));
    std::string part;
    while (std::getline(ss, part, '_')) key.push_back(std::stod(part));
    return key;
}

struct Run {
    long long profit = 0;
    double elapsed = 0;
    long long evaluations = 0;
};

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    try {
        opt = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n" << USAGE;
        return 2;
    }

    const std::string runs_path = opt.out.substr(0, opt.out.size() - 4) + "_runs.csv";
    for (const auto& path : {opt.out, runs_path})
        if (fs::exists(path) && !opt.overwrite) {
            std::cerr << "error: " << path << " exists; choose another --out or pass --overwrite\n";
            return 2;
        }

    const std::vector<std::string> folders = {"SUKP_instances_60/Instances of Set I",
                                              "SUKP_instances_60/Instances of Set II"};
    for (const auto& folder : folders)
        if (!fs::is_directory(folder)) {
            std::cerr << "error: " << folder << " not found; run mmas_run from the repo root\n";
            return 2;
        }
    std::vector<std::string> names = opt.instances;
    if (names.empty()) {
        for (const auto& entry : fs::directory_iterator(folders[opt.set - 1]))
            if (entry.path().extension() == ".txt") names.push_back(entry.path().stem().string());
    }
    std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) { return name_key(a) < name_key(b); });

    std::vector<Instance> insts;
    try {
        for (const auto& name : names) {
            // Instance names are unique across the two sets
            auto folder = std::find_if(folders.begin(), folders.end(), [&](const std::string& f) {
                return fs::exists(f + "/" + name + ".txt");
            });
            if (folder == folders.end()) throw std::runtime_error("instance " + name + " not found in Set I or Set II");
            insts.push_back(load_instance(*folder + "/" + name + ".txt"));
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }

    // Largest instances first, so the slow runs do not end up alone at the tail
    std::vector<std::pair<int, int>> jobs;  // (instance, run)
    std::vector<int> by_size(names.size());
    std::iota(by_size.begin(), by_size.end(), 0);
    std::stable_sort(by_size.begin(), by_size.end(), [&](int a, int b) {
        return (long long)insts[a].m * insts[a].n > (long long)insts[b].m * insts[b].n;
    });
    for (int i : by_size)
        for (int r = 0; r < opt.runs; ++r) jobs.emplace_back(i, r);

    std::vector<std::vector<Run>> runs(names.size(), std::vector<Run>(opt.runs));
    std::vector<std::atomic<int>> finished(names.size());
    std::atomic<size_t> next{0};
    std::mutex io;
    std::string error;
    auto t0 = std::chrono::steady_clock::now();

    auto worker = [&]() {
        while (true) {
            size_t j = next++;
            if (j >= jobs.size()) return;
            auto [i, r] = jobs[j];
            const Instance& inst = insts[i];
            try {
                aco::MMASParams p = opt.mmas;
                p.seed = opt.seed + r;
                if (opt.iter_factor > 0)
                    p.max_iter = std::max(1, (int)std::lround(opt.iter_factor * std::max(inst.m, inst.n)));
                aco::Result res = aco::mmas(inst, p);
                auto [profit, weight] = evaluate(inst, res.solution);
                if (profit != res.profit || weight > inst.capacity)
                    throw std::runtime_error("invalid solution on " + names[i]);
                runs[i][r] = {profit, res.elapsed_sec, res.evaluations};
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(io);
                error = e.what();
                next = jobs.size();
                return;
            }
            if (++finished[i] == opt.runs) {
                std::lock_guard<std::mutex> lock(io);
                double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("%s done (%.0fs)\n", names[i].c_str(), t);
                std::fflush(stdout);
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < opt.threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    if (!error.empty()) {
        std::cerr << "error: " << error << "\n";
        return 1;
    }

    // Best known value: the best result published by the compared literature
    std::map<std::string, long long> bks;
    for (auto& row : read_csv("results_csv/profit.csv"))
        for (const char* col : {"i2pls", "kbts", "e2ls", "ql", "gbpso", "aes_bm"})
            if (!row[col].empty()) bks[row["instance"]] = std::max(bks[row["instance"]], std::stoll(row[col]));

    fs::create_directories(fs::path(opt.out).parent_path());
    std::ofstream summary(opt.out);
    std::ofstream per_run(runs_path);
    summary << "instance,best,mean,std,time,evals,bks,gap_best_%,gap_mean_%\n";
    per_run << "instance,run,seed,profit,time,evals\n";

    std::printf("\nmmas n_ants=%d max_iter=%d iter_factor=%g alpha=%g beta=%g rho=%g tau_min=%g heuristic=%s "
                "gb_every=%d restart=%d, %d runs\n", opt.mmas.n_ants, opt.mmas.max_iter, opt.iter_factor,
                opt.mmas.alpha, opt.mmas.beta, opt.mmas.rho, opt.mmas.tau_min,
                opt.mmas.dynamic ? "dynamic" : "static", opt.mmas.gb_every, opt.mmas.restart, opt.runs);
    std::printf("(max_iter=0 and iter_factor=0: max(m, n); gaps are relative to the best known value BKS)\n");
    std::printf("%-24s%7s%10s%8s%8s | %6s%9s%9s\n", "instance", "best", "mean", "std", "time", "BKS",
                "gap best", "gap mean");
    double sum_gap_best = 0, sum_gap_mean = 0;
    int with_bks = 0, hits = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        const std::string file = names[i] + ".txt";
        double sum = 0, sum_t = 0, sum_e = 0;
        long long best = 0;
        for (int r = 0; r < opt.runs; ++r) {
            const Run& run = runs[i][r];
            sum += run.profit;
            sum_t += run.elapsed;
            sum_e += run.evaluations;
            best = std::max(best, run.profit);
            per_run << names[i] << ',' << r << ',' << opt.seed + r << ',' << run.profit << ',' << run.elapsed
                    << ',' << run.evaluations << '\n';
        }
        double mean = sum / opt.runs, var = 0;
        for (int r = 0; r < opt.runs; ++r) var += (runs[i][r].profit - mean) * (runs[i][r].profit - mean);
        double sd = std::sqrt(var / opt.runs);
        double t = sum_t / opt.runs;

        char bk[16] = "-", gap_best[16] = "-", gap_mean[16] = "-";
        std::string bk_csv, gap_best_csv, gap_mean_csv;
        if (bks.count(file)) {
            double gb = 100.0 * (bks[file] - best) / bks[file], gm = 100.0 * (bks[file] - mean) / bks[file];
            std::snprintf(bk, sizeof bk, "%lld", bks[file]);
            std::snprintf(gap_best, sizeof gap_best, "%.2f", gb);
            std::snprintf(gap_mean, sizeof gap_mean, "%.2f", gm);
            bk_csv = bk;
            gap_best_csv = gap_best;
            gap_mean_csv = gap_mean;
            sum_gap_best += gb;
            sum_gap_mean += gm;
            ++with_bks;
            hits += best >= bks[file];
        }
        summary << names[i] << ',' << best << ',' << mean << ',' << sd << ',' << t << ',' << (long long)(sum_e / opt.runs)
                << ',' << bk_csv << ',' << gap_best_csv << ',' << gap_mean_csv << '\n';
        std::printf("%-24s%7lld%10.1f%8.1f%8.3f | %6s%9s%9s\n", names[i].c_str(), best, mean, sd, t, bk, gap_best,
                    gap_mean);
    }
    if (with_bks > 0)
        std::printf("average over %d instances: gap best %.2f%%, gap mean %.2f%%, BKS reached on %d\n", with_bks,
                    sum_gap_best / with_bks, sum_gap_mean / with_bks, hits);
    std::printf("saved to %s and %s\n", opt.out.c_str(), runs_path.c_str());
    return 0;
}
