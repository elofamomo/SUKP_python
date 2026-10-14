#pragma once
#include "instance.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <stdexcept>
#include <cstdint>

// SUKP core: the incremental knapsack state (Cover), the greedy ratio p_i / R_i of He et al.
// (2018) used as the ants' heuristic, and the result of one run.
namespace aco {

// Instance plus the element -> items index (CSR), so covering an element updates exactly
// the items that hold it.
struct Problem {
    const Instance& inst;
    std::vector<int> elem_offset;        // size n+1
    std::vector<int> elem_items;         // items holding each element, concatenated
    std::vector<long long> item_weight;  // weight of each item alone

    explicit Problem(const Instance& instance)
        : inst(instance), elem_offset(instance.n + 1, 0), item_weight(instance.m, 0) {
        for (int k = 0; k < inst.subset_offset[inst.m]; ++k) ++elem_offset[inst.subset_flat[k] + 1];
        for (int j = 0; j < inst.n; ++j) elem_offset[j + 1] += elem_offset[j];
        elem_items.resize(elem_offset[inst.n]);
        std::vector<int> pos(elem_offset.begin(), elem_offset.end() - 1);
        for (int i = 0; i < inst.m; ++i)
            for (int k = inst.subset_offset[i]; k < inst.subset_offset[i + 1]; ++k) {
                int e = inst.subset_flat[k];
                elem_items[pos[e]++] = i;
                item_weight[i] += inst.weight[e];
            }
    }

    int degree(int e) const { return elem_offset[e + 1] - elem_offset[e]; }
};

// Items chosen so far and the union of their elements. gain[i] is the weight item i would add
// if chosen now; covering element e lowers the gain of every item holding e, so one whole
// construction costs O(nonzeros of the relation matrix).
// load + gain[i] never decreases as items are added, so an item that does not fit now never will.
class Cover {
public:
    const Problem& pb;
    std::vector<uint8_t> selected, covered;
    std::vector<long long> gain;
    long long load = 0, profit = 0;

    explicit Cover(const Problem& p)
        : pb(p), selected(p.inst.m, 0), covered(p.inst.n, 0), gain(p.item_weight) {}

    void reset() {
        std::fill(selected.begin(), selected.end(), 0);
        std::fill(covered.begin(), covered.end(), 0);
        gain = pb.item_weight;
        load = profit = 0;
    }

    bool fits(int i) const { return load + gain[i] <= pb.inst.capacity; }

    // Selects item i (the caller checks fits(i)).
    void add(int i) {
        const Instance& inst = pb.inst;
        selected[i] = 1;
        profit += inst.profit[i];
        for (int k = inst.subset_offset[i]; k < inst.subset_offset[i + 1]; ++k) {
            int e = inst.subset_flat[k];
            if (covered[e]) continue;
            covered[e] = 1;
            long long w = inst.weight[e];
            load += w;
            for (int t = pb.elem_offset[e]; t < pb.elem_offset[e + 1]; ++t) gain[pb.elem_items[t]] -= w;
        }
    }
};

// p_i / R_i with R_i = sum_{j in U_i} w_j / d_j (He et al. 2018, Sec. 4.1); +inf for an item with no element.
inline std::vector<double> ratio(const Problem& pb) {
    const Instance& inst = pb.inst;
    std::vector<double> share(inst.n, 0.0);
    for (int j = 0; j < inst.n; ++j)
        if (pb.degree(j) > 0) share[j] = (double)inst.weight[j] / pb.degree(j);
    std::vector<double> r(inst.m);
    for (int i = 0; i < inst.m; ++i) {
        double R = 0.0;
        for (int k = inst.subset_offset[i]; k < inst.subset_offset[i + 1]; ++k) R += share[inst.subset_flat[k]];
        r[i] = R > 0 ? (double)inst.profit[i] / R : std::numeric_limits<double>::infinity();
    }
    return r;
}

struct Result {
    std::vector<uint8_t> solution;
    long long profit = -1;
    std::vector<long long> history;  // best profit after each iteration
    long long evaluations = 0;       // solutions built (ants for MMAS)
    double elapsed_sec = 0;
};

}  // namespace aco
