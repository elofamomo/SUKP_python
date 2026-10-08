#pragma once
#include "instance.hpp"
#include <vector>
#include <random>
#include <unordered_set>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>

// Incremental SUKP state, ported from helper/set_handler.py's SetUnionHandler:
// element_counts tracks how many selected items currently cover each element,
// so add/remove cost O(|item's subset|) instead of O(n) recompute.
class SUKPState {
public:
    const Instance& inst;
    std::vector<uint8_t> selected;        // size m
    std::vector<int32_t> element_counts;  // size n
    long long total_profit = 0;
    long long total_weight = 0;
    uint64_t hash = 0;                    // Zobrist-style incremental hash of the selected set
    std::vector<uint64_t> item_hash;      // size m, precomputed per-item random values

    explicit SUKPState(const Instance& instance, uint64_t seed)
        : inst(instance), selected(instance.m, 0), element_counts(instance.n, 0) {
        std::mt19937_64 rng(seed);
        item_hash.resize(inst.m);
        for (auto& h : item_hash) h = rng();
    }

    inline int subset_begin(int item) const { return inst.subset_offset[item]; }
    inline int subset_end(int item) const { return inst.subset_offset[item + 1]; }

    // AW(S,i) for a not-yet-selected item: weight newly covered if added.
    long long additional_weight_add(int item) const {
        long long w = 0;
        for (int k = subset_begin(item); k < subset_end(item); ++k) {
            int e = inst.subset_flat[k];
            if (element_counts[e] == 0) w += inst.weight[e];
        }
        return w;
    }

    // AW(S,i) for a selected item: weight that would be freed if removed
    // (elements only this item currently covers).
    long long additional_weight_remove(int item) const {
        long long w = 0;
        for (int k = subset_begin(item); k < subset_end(item); ++k) {
            int e = inst.subset_flat[k];
            if (element_counts[e] == 1) w += inst.weight[e];
        }
        return w;
    }

    bool add_item(int item) {
        if (selected[item]) return false;
        long long add_w = additional_weight_add(item);
        if (total_weight + add_w > inst.capacity) return false;
        for (int k = subset_begin(item); k < subset_end(item); ++k) {
            int e = inst.subset_flat[k];
            if (element_counts[e]++ == 0) total_weight += inst.weight[e];
        }
        selected[item] = 1;
        total_profit += inst.profit[item];
        hash ^= item_hash[item];
        return true;
    }

    bool remove_item(int item) {
        if (!selected[item]) return false;
        for (int k = subset_begin(item); k < subset_end(item); ++k) {
            int e = inst.subset_flat[k];
            if (--element_counts[e] == 0) total_weight -= inst.weight[e];
        }
        selected[item] = 0;
        total_profit -= inst.profit[item];
        hash ^= item_hash[item];
        return true;
    }

    void reset() {
        std::fill(selected.begin(), selected.end(), 0);
        std::fill(element_counts.begin(), element_counts.end(), 0);
        total_profit = 0;
        total_weight = 0;
        hash = 0;
    }
};

struct E2LSParams {
    int rnum = 2;                          // # worst-ratio selected items considered for removal
    int anum = 2;                          // # best-ratio candidates branched on per ADD* node
    int construct_t = -1;                  // Random_Greedy sample size; -1 => sqrt(max(m,n))
    long long add_star_node_budget = 4000;  // caps ADD* branching cost per removal candidate
    double time_budget_sec = 30.0;
    uint64_t seed = 1;
};

struct E2LSResult {
    std::vector<uint8_t> best_selected;
    long long best_profit = -1;
    double elapsed_sec = 0;
    long long iterations = 0;
    long long restarts = 0;
};

// C++ port of the E2LS-style local search (Wei & Hao-family "restricted
// candidate + recursive multi-item add-back" local search for SUKP), per
// the algorithm summary distilled from arXiv:2207.00749. The exact
// backtracking/threading semantics of the paper's ADD* were ambiguous in
// the extracted pseudocode, so this implementation favors a clean, provably
// self-undoing recursive search with the same spirit: restrict which items
// are ever considered (top-rnum worst-ratio for removal, top-anum best-ratio
// for addition), always take free (zero-marginal-weight) items unconditionally,
// and track the best feasible snapshot seen anywhere in the search tree.
class E2LSSolver {
public:
    E2LSSolver(const Instance& inst, E2LSParams params)
        : inst_(inst), params_(params), state_(inst, params.seed), rng_(params.seed) {
        if (params_.construct_t <= 0) {
            int mx = std::max(inst_.m, inst_.n);
            params_.construct_t = std::max(1, (int)std::llround(std::sqrt((double)mx)));
        }
    }

    E2LSResult run() {
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        E2LSResult result;

        random_greedy_construct();
        mark_visited();
        record_best_if_improved();

        while (elapsed_seconds(t0) < params_.time_budget_sec) {
            bool moved = local_search_step();
            result.iterations++;
            if (!moved) {
                state_.reset();
                random_greedy_construct();
                mark_visited();
                result.restarts++;
            }
            record_best_if_improved();
        }

        result.elapsed_sec = elapsed_seconds(t0);
        result.best_profit = best_profit_ever_;
        result.best_selected = best_selected_ever_;
        return result;
    }

private:
    const Instance& inst_;
    E2LSParams params_;
    SUKPState state_;
    std::mt19937_64 rng_;
    std::unordered_set<uint64_t> visited_;
    long long best_profit_ever_ = -1;
    std::vector<uint8_t> best_selected_ever_;

    static double elapsed_seconds(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    bool is_tabu(uint64_t h) const { return visited_.count(h) != 0; }
    void mark_visited() { visited_.insert(state_.hash); }

    void record_best_if_improved() {
        if (state_.total_profit > best_profit_ever_) {
            best_profit_ever_ = state_.total_profit;
            best_selected_ever_ = state_.selected;
        }
    }

    static double ratio(long long profit, long long aw) {
        if (aw <= 0) return std::numeric_limits<double>::infinity();
        return (double)profit / (double)aw;
    }

    uint64_t hash_of(const std::vector<uint8_t>& sel) const {
        uint64_t h = 0;
        for (int i = 0; i < inst_.m; ++i)
            if (sel[i]) h ^= state_.item_hash[i];
        return h;
    }

    // Adds every currently-free (AW==0) item, tracking what it applied so the
    // caller can undo it later; free items can cascade (adding one may free
    // another), so this loops until a fixed point.
    void add_all_free_items(std::vector<int>& applied) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < inst_.m; ++i) {
                if (state_.selected[i]) continue;
                if (state_.additional_weight_add(i) == 0) {
                    state_.add_item(i);
                    applied.push_back(i);
                    changed = true;
                }
            }
        }
    }

    // Recursive bounded-branching add-back search. Fully undoes every change
    // it makes before returning (invariant: state_ is unchanged across a call),
    // while recording the best feasible snapshot seen anywhere in the subtree
    // into local_best_profit/local_best_selected.
    void add_star(long long& node_budget, long long& local_best_profit,
                  std::vector<uint8_t>& local_best_selected) {
        std::vector<int> applied_free;
        add_all_free_items(applied_free);
        if (state_.total_profit > local_best_profit) {
            local_best_profit = state_.total_profit;
            local_best_selected = state_.selected;
        }

        if (node_budget > 0) {
            std::vector<std::pair<int, double>> fc; // (item, ratio)
            fc.reserve(inst_.m);
            for (int i = 0; i < inst_.m; ++i) {
                if (state_.selected[i]) continue;
                long long aw = state_.additional_weight_add(i);
                if (aw > 0 && state_.total_weight + aw <= inst_.capacity)
                    fc.emplace_back(i, ratio(inst_.profit[i], aw));
            }
            std::sort(fc.begin(), fc.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

            int branch = std::min<int>(params_.anum, (int)fc.size());
            for (int b = 0; b < branch && node_budget > 0; ++b) {
                int cand = fc[b].first;
                --node_budget;
                state_.add_item(cand);
                if (state_.total_profit > local_best_profit) {
                    local_best_profit = state_.total_profit;
                    local_best_selected = state_.selected;
                }
                add_star(node_budget, local_best_profit, local_best_selected);
                state_.remove_item(cand);
            }
        }

        for (auto it = applied_free.rbegin(); it != applied_free.rend(); ++it)
            state_.remove_item(*it);
    }

    void random_greedy_construct() {
        state_.reset();
        std::vector<int> scratch; // unused undo-tracking for the top-level free pass
        while (true) {
            add_all_free_items(scratch);
            scratch.clear();

            std::vector<int> feasible;
            feasible.reserve(inst_.m);
            for (int i = 0; i < inst_.m; ++i) {
                if (state_.selected[i]) continue;
                long long aw = state_.additional_weight_add(i);
                if (aw > 0 && state_.total_weight + aw <= inst_.capacity) feasible.push_back(i);
            }
            if (feasible.empty()) break;

            int t = std::min<int>(params_.construct_t, (int)feasible.size());
            std::uniform_int_distribution<size_t> dist(0, feasible.size() - 1);
            int best_item = -1;
            double best_r = -1;
            for (int s = 0; s < t; ++s) {
                int cand = feasible[dist(rng_)];
                double r = ratio(inst_.profit[cand], state_.additional_weight_add(cand));
                if (r > best_r) { best_r = r; best_item = cand; }
            }
            state_.add_item(best_item);
        }
    }

    // Replays a target selection onto state_ via incremental add/remove.
    // Correct regardless of order: removing extras first can only shrink
    // weight, and replaying a subset of the target's items can never exceed
    // the target's own (already-feasible) total weight.
    void apply_selection(const std::vector<uint8_t>& target) {
        for (int i = 0; i < inst_.m; ++i)
            if (state_.selected[i] && !target[i]) state_.remove_item(i);
        for (int i = 0; i < inst_.m; ++i)
            if (!state_.selected[i] && target[i]) state_.add_item(i);
    }

    bool local_search_step() {
        std::vector<int> in_sol;
        in_sol.reserve(inst_.m);
        for (int i = 0; i < inst_.m; ++i)
            if (state_.selected[i]) in_sol.push_back(i);

        if (in_sol.empty()) {
            long long budget = params_.add_star_node_budget;
            long long local_best_profit = state_.total_profit;
            std::vector<uint8_t> local_best_selected = state_.selected;
            add_star(budget, local_best_profit, local_best_selected);
            if (local_best_profit > state_.total_profit) {
                apply_selection(local_best_selected);
                mark_visited();
                return true;
            }
            return false;
        }

        std::sort(in_sol.begin(), in_sol.end(), [&](int a, int b) {
            return ratio(inst_.profit[a], state_.additional_weight_remove(a)) <
                   ratio(inst_.profit[b], state_.additional_weight_remove(b));
        });
        int rcount = std::min<int>(params_.rnum, (int)in_sol.size());

        long long best_alt_profit = -1;
        std::vector<uint8_t> best_alt_selected;

        for (int r = 0; r < rcount; ++r) {
            int victim = in_sol[r];
            state_.remove_item(victim);

            long long budget = params_.add_star_node_budget;
            long long local_best_profit = state_.total_profit;
            std::vector<uint8_t> local_best_selected = state_.selected;
            add_star(budget, local_best_profit, local_best_selected);

            bool tabu = is_tabu(hash_of(local_best_selected));
            bool aspiration = local_best_profit > best_profit_ever_;
            if ((!tabu || aspiration) && local_best_profit > best_alt_profit) {
                best_alt_profit = local_best_profit;
                best_alt_selected = local_best_selected;
            }

            state_.add_item(victim); // restore original state for the next candidate
        }

        // Only strict improvement counts as progress; ties/worse are treated
        // as a dead end (triggers a restart in run()) to avoid cycling.
        if (best_alt_selected.empty() || best_alt_profit <= state_.total_profit)
            return false;

        apply_selection(best_alt_selected);
        mark_visited();
        return true;
    }
};
