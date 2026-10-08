#pragma once
#include "instance.hpp"
#include <vector>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <algorithm>
#include <stdexcept>
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

using CandidateScorer = std::function<void(const std::vector<uint8_t>& selected,
                                           const std::vector<int>& candidates,
                                           std::vector<double>& scores)>;

struct SUKPParams {
    int rnum = 2;                          // # worst-ratio selected items considered for removal
    int anum = 2;                          // # best-ranked candidates branched on per add-back node
    int construct_t = -1;                  // random_greedy_construct sample size; -1 => sqrt(max(m,n))
    long long add_back_node_budget = -1;   // caps add-back branches per removal candidate; -1 => no cap (paper)
    double time_budget_sec = 30.0;         // <= 0 => no time limit
    long long max_iterations = -1;         // local search steps; -1 => no limit
    uint64_t seed = 1;
    // Training labels: chance of labelling the add-back from each removal trial by exhaustive search
    double sample_prob = 0.0;
    long long max_samples = -1;            // -1 => no limit
    long long oracle_node_budget = 200000; // per label; labels whose exhaustive search exceeds it are dropped
};


struct AddBackSample {
    std::vector<uint8_t> selected;
    std::vector<int> candidates;
    std::vector<long long> values;
};

struct SUKPResult {
    std::vector<uint8_t> best_selected;
    long long best_profit = -1;
    double elapsed_sec = 0;
    long long iterations = 0;
    long long restarts = 0;
    long long add_back_nodes = 0;          // branches taken across all add-back searches
    long long scorer_calls = 0;
    std::vector<AddBackSample> samples;
};


class SUKPSolver {
public:
    SUKPSolver(const Instance& inst, SUKPParams params)
        : inst_(inst), params_(params), state_(inst, params.seed), rng_(params.seed),
          sample_rng_(params.seed ^ 0x9e3779b97f4a7c15ULL) {
        if (params_.time_budget_sec <= 0 && params_.max_iterations < 0)
            throw std::invalid_argument("set time_budget_sec > 0 or max_iterations >= 0");
        if (params_.construct_t <= 0) {
            int mx = std::max(inst_.m, inst_.n);
            params_.construct_t = std::max(1, (int)std::llround(std::sqrt((double)mx)));
        }
    }

    // Ranks add-back candidates with this instead of the value-weight ratio.
    void set_scorer(CandidateScorer scorer) { scorer_ = std::move(scorer); }
    // Polled once per iteration; returning true ends the run early (Ctrl-C from Python).
    void set_interrupt_check(std::function<bool()> check) { interrupt_check_ = std::move(check); }

    SUKPResult run() {
        auto t0 = std::chrono::steady_clock::now();
        result_ = SUKPResult();

        random_greedy_construct();
        mark_visited();
        record_best_if_improved();

        while (!budget_exhausted(t0)) {
            if (interrupt_check_ && interrupt_check_()) break;
            bool moved = local_search_step();
            result_.iterations++;
            if (!moved) {
                random_greedy_construct();
                mark_visited();
                result_.restarts++;
            }
            record_best_if_improved();
        }

        result_.elapsed_sec = elapsed_seconds(t0);
        result_.best_profit = best_profit_ever_;
        result_.best_selected = best_selected_ever_;
        return std::move(result_);
    }

private:
    const Instance& inst_;
    SUKPParams params_;
    SUKPState state_;
    std::mt19937_64 rng_;
    std::mt19937_64 sample_rng_;  // separate stream, so labelling does not change the search
    std::unordered_set<uint64_t> visited_;
    long long best_profit_ever_ = -1;
    std::vector<uint8_t> best_selected_ever_;
    CandidateScorer scorer_;
    std::function<bool()> interrupt_check_;
    SUKPResult result_;

    static double elapsed_seconds(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    bool budget_exhausted(std::chrono::steady_clock::time_point t0) const {
        if (params_.max_iterations >= 0 && result_.iterations >= params_.max_iterations) return true;
        return params_.time_budget_sec > 0 && elapsed_seconds(t0) >= params_.time_budget_sec;
    }

    bool is_tabu(uint64_t h) const { return visited_.count(h) != 0; }
    bool is_tabu_with(int item) const { return is_tabu(state_.hash ^ state_.item_hash[item]); }
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

    // Adds every currently-free (AW==0) item, appending each to applied so the caller can undo
    // it; adding one can free another, so this loops to a fixed point. With respect_tabu, items
    // whose addition gives a tabu solution are skipped (paper Algorithm 3, lines 5-8).
    void add_free_items(std::vector<int>& applied, bool respect_tabu) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < inst_.m; ++i) {
                if (state_.selected[i]) continue;
                if (state_.additional_weight_add(i) != 0) continue;
                if (respect_tabu && is_tabu_with(i)) continue;
                state_.add_item(i);
                applied.push_back(i);
                changed = true;
            }
        }
    }

    void undo_additions(const std::vector<int>& applied) {
        for (auto it = applied.rbegin(); it != applied.rend(); ++it)
            state_.remove_item(*it);
    }

    // The feasible candidates FC (AW > 0 and fits), best first: by value-weight ratio (paper),
    // or by the scorer when one is set and there are more than anum to choose from.
    std::vector<int> ranked_candidates(bool use_scorer) {
        std::vector<std::pair<double, int>> fc;  // (ratio, item)
        for (int i = 0; i < inst_.m; ++i) {
            if (state_.selected[i]) continue;
            long long aw = state_.additional_weight_add(i);
            if (aw > 0 && state_.total_weight + aw <= inst_.capacity)
                fc.emplace_back(ratio(inst_.profit[i], aw), i);
        }
        std::stable_sort(fc.begin(), fc.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<int> items;
        items.reserve(fc.size());
        for (const auto& p : fc) items.push_back(p.second);

        if (use_scorer && scorer_ && (int)items.size() > params_.anum) {
            std::vector<double> scores(items.size());
            scorer_(state_.selected, items, scores);
            result_.scorer_calls++;
            std::vector<int> order(items.size());
            for (size_t k = 0; k < order.size(); ++k) order[k] = (int)k;
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return scores[a] > scores[b]; });
            std::vector<int> ranked;
            ranked.reserve(items.size());
            for (int k : order) ranked.push_back(items[k]);
            items.swap(ranked);
        }
        return items;
    }

    void offer_step_best(long long& step_best_profit, std::vector<uint8_t>& step_best_selected) const {
        if (state_.total_profit > step_best_profit) {
            step_best_profit = state_.total_profit;
            step_best_selected = state_.selected;
        }
    }

    // ADD* (paper Algorithm 3): adds the free items, then branches on the anum best-ranked
    // candidates (skipping tabu ones), recursing until nothing fits. Every non-tabu solution it
    // reaches is offered to the step's best. state_ is unchanged across a call.
    void add_back_search(long long& node_budget, long long& step_best_profit,
                         std::vector<uint8_t>& step_best_selected) {
        std::vector<int> applied_free;
        add_free_items(applied_free, /*respect_tabu=*/true);
        if (!applied_free.empty()) offer_step_best(step_best_profit, step_best_selected);

        std::vector<int> fc = ranked_candidates(/*use_scorer=*/true);
        int branch = std::min<int>(params_.anum, (int)fc.size());
        for (int b = 0; b < branch && node_budget != 0; ++b) {
            int cand = fc[b];
            if (is_tabu_with(cand)) continue;
            if (node_budget > 0) --node_budget;
            result_.add_back_nodes++;
            state_.add_item(cand);
            offer_step_best(step_best_profit, step_best_selected);
            add_back_search(node_budget, step_best_profit, step_best_selected);
            state_.remove_item(cand);
        }

        undo_additions(applied_free);
    }

    void random_greedy_construct() {
        state_.reset();
        std::vector<int> scratch; // unused undo-tracking for the top-level free pass
        while (true) {
            add_free_items(scratch, /*respect_tabu=*/false);
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

    // Local_Search (paper Algorithm 2): removes each of the rnum lowest-ratio items in turn and
    // re-fills with ADD*. The pseudocode's tabu test on S\{i} (line 7) is ambiguous; following
    // the text, ADD* always runs and only the solutions it reaches are candidates. Returns false
    // when all of them are tabu (the caller restarts); otherwise moves to the best one.
    bool local_search_step() {
        std::vector<std::pair<double, int>> removal;  // (ratio, item)
        for (int i = 0; i < inst_.m; ++i)
            if (state_.selected[i])
                removal.emplace_back(ratio(inst_.profit[i], state_.additional_weight_remove(i)), i);
        std::stable_sort(removal.begin(), removal.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        int rcount = std::min<int>(params_.rnum, (int)removal.size());

        long long step_best_profit = -1;
        std::vector<uint8_t> step_best_selected;

        for (int r = 0; r < rcount; ++r) {
            int victim = removal[r].second;
            state_.remove_item(victim);
            if (should_sample()) record_samples();

            long long budget = params_.add_back_node_budget;
            add_back_search(budget, step_best_profit, step_best_selected);

            state_.add_item(victim); // restore original state for the next candidate
        }

        if (step_best_profit < 0) return false;

        apply_selection(step_best_selected);
        mark_visited();
        return true;
    }

    bool should_sample() {
        if (params_.sample_prob <= 0) return false;
        if (params_.max_samples >= 0 && (long long)result_.samples.size() >= params_.max_samples)
            return false;
        return std::uniform_real_distribution<double>(0.0, 1.0)(sample_rng_) < params_.sample_prob;
    }

    // Best profit reachable from state_ by additions only (free items, then any feasible
    // candidates), ignoring tabu; memoized on the selection hash. If the node budget runs out it
    // sets truncated and the value is only a lower bound. state_ is unchanged on return.
    long long exact_completion(long long& budget, bool& truncated,
                               std::unordered_map<uint64_t, long long>& memo) {
        auto it = memo.find(state_.hash);
        if (it != memo.end()) return it->second;
        if (budget-- <= 0) { truncated = true; return state_.total_profit; }
        uint64_t key = state_.hash;

        std::vector<int> applied;
        add_free_items(applied, /*respect_tabu=*/false);
        long long best = state_.total_profit;
        for (int cand : ranked_candidates(/*use_scorer=*/false)) {
            state_.add_item(cand);
            best = std::max(best, exact_completion(budget, truncated, memo));
            state_.remove_item(cand);
            if (truncated) break;
        }
        undo_additions(applied);

        if (!truncated) memo.emplace(key, best);
        return best;
    }

    // Labels the add-back decisions along the exact-optimal path from the current state (S minus
    // the removed item), appending one sample per node with at least two candidates. Stops at
    // the first node whose exhaustive search exceeds the budget. state_ is unchanged on return.
    void record_samples() {
        std::unordered_map<uint64_t, long long> memo;
        std::vector<int> applied;  // every item added on the way down, undone at the end
        while (params_.max_samples < 0 || (long long)result_.samples.size() < params_.max_samples) {
            add_free_items(applied, /*respect_tabu=*/false);
            std::vector<int> fc = ranked_candidates(/*use_scorer=*/false);
            if (fc.size() < 2) break;

            AddBackSample sample{state_.selected, fc, {}};
            long long budget = params_.oracle_node_budget;
            bool truncated = false;
            for (int cand : fc) {
                state_.add_item(cand);
                sample.values.push_back(exact_completion(budget, truncated, memo));
                state_.remove_item(cand);
                if (truncated) break;
            }
            if (truncated) break;

            size_t best = std::max_element(sample.values.begin(), sample.values.end()) - sample.values.begin();
            int best_item = fc[best];
            result_.samples.push_back(std::move(sample));
            state_.add_item(best_item);
            applied.push_back(best_item);
        }
        undo_additions(applied);
    }
};
