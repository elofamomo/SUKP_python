#pragma once
#include "sukp.hpp"

// Baseline: binary artificial bee colony of He, Xie, Wong & Wang, "A novel binary artificial bee
// colony algorithm for the set-union knapsack problem", FGCS 78 (2018), Algorithm 3.
namespace aco {

// H[1..m]: items sorted by p_i / R_i in descending order (Eq. (10)).
inline std::vector<int> ratio_order(const Problem& pb) {
    std::vector<double> r = ratio(pb);
    std::vector<int> order(pb.inst.m);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return r[a] > r[b]; });
    return order;
}

// S-GROA (Algorithm 2), result left in cover. Repairing stage, only if y is infeasible: rebuild
// from empty in H order keeping the items of y that fit. Optimizing stage: add, in H order,
// every item that still fits.
inline void sgroa(Cover& cover, const std::vector<int>& order, const std::vector<uint8_t>& y) {
    const Instance& inst = cover.pb.inst;
    cover.reset();
    for (int i = 0; i < inst.m; ++i)
        if (y[i]) cover.add(i);
    if (cover.load > inst.capacity) {
        cover.reset();
        for (int i : order)
            if (y[i] && cover.fits(i)) cover.add(i);
    }
    for (int i : order)
        if (!cover.selected[i] && cover.fits(i)) cover.add(i);
}

struct BABCParams {
    int n_bees = 20;     // N
    int max_iter = 0;    // MIT; <= 0 => max(m, n)
    double a = 5.0;      // search space [-a, a]^m
    double limit = 0;    // <= 0 => max(m, n) / 5
    uint64_t seed = 0;
};

// A food source is a real vector X in [-a, a]^m, mapped to items by sign (y_i = 1 iff x_i >= 0,
// Eq. (9)); S-GROA repairs and fills it, and the fitness of X is the profit of that solution.
// Two details the paper leaves open are decided as follows:
//   * Eq. (6) moves every dimension, each with its own phi in [-1, 1] (the "full mapping"),
//     and V is clipped back into [-a, a]^m.
//   * The onlooker phase follows Algorithm 3 step 4.2 literally: bee i searches once if
//     p_i > rand(0, 1), with p_i = fit_i / sum(fit) (Eq. (7)).
inline Result babc(const Instance& inst, const BABCParams& prm) {
    auto t0 = std::chrono::steady_clock::now();
    if (prm.n_bees < 2) throw std::invalid_argument("babc needs at least 2 bees");
    Problem pb(inst);
    const int m = inst.m, N = prm.n_bees;
    const int max_iter = prm.max_iter > 0 ? prm.max_iter : std::max(inst.m, inst.n);
    const double limit = prm.limit > 0 ? prm.limit : std::max(inst.m, inst.n) / 5.0;
    const double a = prm.a;
    std::mt19937_64 rng(prm.seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::uniform_int_distribution<int> partner(0, N - 2);
    const std::vector<int> order = ratio_order(pb);
    Cover cover(pb);
    std::vector<uint8_t> y(m);
    Result res;

    // Eq. (9) mapping, S-GROA, fitness = profit of the feasible solution
    auto evaluate = [&](const double* x) {
        for (int j = 0; j < m; ++j) y[j] = x[j] >= 0;
        sgroa(cover, order, y);
        ++res.evaluations;
        if (cover.profit > res.profit) {
            res.profit = cover.profit;
            res.solution = cover.selected;
        }
        return (double)cover.profit;
    };

    std::vector<double> X((size_t)N * m), V(m), fit(N), p(N);
    std::vector<int> trial(N, 0);
    for (auto& v : X) v = -a + 2 * a * U(rng);  // Eq. (8)
    for (int i = 0; i < N; ++i) fit[i] = evaluate(&X[(size_t)i * m]);

    // Eq. (6) around food source i, every dimension, partner k != i, then greedy selection
    auto search = [&](int i) {
        int k = partner(rng);
        k += k >= i;
        const double* xi = &X[(size_t)i * m];
        const double* xk = &X[(size_t)k * m];
        for (int j = 0; j < m; ++j) {
            double phi = -1 + 2 * U(rng);
            V[j] = std::clamp(xi[j] + phi * (xi[j] - xk[j]), -a, a);
        }
        double f = evaluate(V.data());
        if (f > fit[i]) {
            std::copy(V.begin(), V.end(), X.begin() + (size_t)i * m);
            fit[i] = f;
            trial[i] = 0;
        } else {
            ++trial[i];
        }
    };

    for (int t = 0; t < max_iter; ++t) {
        for (int i = 0; i < N; ++i) search(i);  // employed bees
        double total = std::accumulate(fit.begin(), fit.end(), 0.0);
        for (int i = 0; i < N; ++i) p[i] = total > 0 ? fit[i] / total : 1.0 / N;  // Eq. (7)
        for (int i = 0; i < N; ++i)  // onlooker bees, Algorithm 3 step 4.2
            if (p[i] > U(rng)) search(i);
        for (int i = 0; i < N; ++i)  // scout bees
            if (trial[i] > limit) {
                double* xi = &X[(size_t)i * m];
                for (int j = 0; j < m; ++j) xi[j] = -a + 2 * a * U(rng);
                fit[i] = evaluate(xi);
                trial[i] = 0;
            }
        res.history.push_back(res.profit);
    }
    res.elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

}  // namespace aco
