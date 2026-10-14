#pragma once
#include "sukp.hpp"

// MAX-MIN Ant System for the SUKP (Stutzle & Hoos 2000), with one pheromone value per item.
namespace aco {

struct MMASParams {
    int n_ants = 20;
    int max_iter = 0;      // <= 0 => max(m, n)
    double alpha = 1.0;    // pheromone exponent
    double beta = 2.0;     // heuristic exponent
    double rho = 0.1;      // evaporation rate
    double tau_min = 0.01; // the upper bound is 1
    bool dynamic = false;  // eta_i = p_i / gain_i instead of p_i / R_i
    int gb_every = 0;      // every gb_every-th iteration the global best deposits instead of the iteration best; 0: never
    int restart = 0;       // reset tau to 1 after this many iterations without a new global best; 0: never
    uint64_t seed = 0;
};

// x^k for a small non-negative integer k by repeated squaring, much faster than std::pow
inline double ipow(double x, int k) {
    double r = 1.0;
    for (; k > 0; k >>= 1, x *= x)
        if (k & 1) r *= x;
    return r;
}

// One ant: from an empty knapsack, repeatedly pick one item among those that still fit with
// probability proportional to tau_i^alpha * eta_i^beta, until nothing fits. Items that add no
// weight are taken at once (this does not change the distribution of the final set).
// beta_int >= 0 when beta is that integer (the dynamic heuristic then uses ipow).
// cand and cum are scratch buffers reused across ants.
inline void construct(Cover& cover, const std::vector<double>& tau_a, const std::vector<double>& attract,
                      bool dynamic, double beta, int beta_int, std::mt19937_64& rng, std::vector<int>& cand,
                      std::vector<double>& cum) {
    const Instance& inst = cover.pb.inst;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    cover.reset();
    cand.resize(inst.m);
    std::iota(cand.begin(), cand.end(), 0);
    while (true) {
        // Drop items taken or no longer fitting (they never fit again); take free items now
        size_t kept = 0;
        for (size_t c = 0; c < cand.size(); ++c) {
            int i = cand[c];
            if (cover.selected[i] || !cover.fits(i)) continue;
            if (cover.gain[i] == 0) {
                cover.add(i);
                continue;
            }
            cand[kept++] = i;
        }
        cand.resize(kept);
        if (kept == 0) return;
        cum.resize(kept);
        double total = 0.0;
        for (size_t c = 0; c < kept; ++c) {
            int i = cand[c];
            if (dynamic) {
                double eta = (double)inst.profit[i] / cover.gain[i];
                total += tau_a[i] * (beta_int >= 0 ? ipow(eta, beta_int) : std::pow(eta, beta));
            } else {
                total += attract[i];
            }
            cum[c] = total;
        }
        size_t pick;
        if (total > 0) {
            pick = std::upper_bound(cum.begin(), cum.end(), U(rng) * total) - cum.begin();
            pick = std::min(pick, kept - 1);
        } else {
            pick = std::uniform_int_distribution<size_t>(0, kept - 1)(rng);
        }
        cover.add(cand[pick]);
    }
}

// MAX-MIN Ant System with one pheromone value per item and the hyper-cube update by the
// iteration-best ant: tau <- (1 - rho) * tau + rho * [i in best], clamped to [tau_min, 1].
// Optionally, as in the original MMAS, the global best deposits every gb_every-th iteration
// and the pheromone is reset when the search stagnates.
inline Result mmas(const Instance& inst, const MMASParams& prm) {
    auto t0 = std::chrono::steady_clock::now();
    Problem pb(inst);
    const int m = inst.m;
    const int max_iter = prm.max_iter > 0 ? prm.max_iter : std::max(inst.m, inst.n);
    std::mt19937_64 rng(prm.seed);
    const int beta_int = prm.beta >= 0 && prm.beta <= 64 && prm.beta == std::floor(prm.beta) ? (int)prm.beta : -1;
    std::vector<double> eta = ratio(pb);
    for (auto& e : eta)
        if (std::isinf(e)) e = 0.0;  // an item with no element is free and taken at once
    std::vector<double> tau(m, 1.0), tau_a(m), attract(m), cum;
    std::vector<int> cand;
    std::vector<uint8_t> it_sel;
    Cover cover(pb);
    Result res;
    int stale = 0;  // iterations since the global best last improved

    for (int t = 0; t < max_iter; ++t) {
        for (int i = 0; i < m; ++i) {
            tau_a[i] = std::pow(tau[i], prm.alpha);
            attract[i] = tau_a[i] * std::pow(eta[i], prm.beta);
        }
        long long it_profit = -1;
        for (int k = 0; k < prm.n_ants; ++k) {
            construct(cover, tau_a, attract, prm.dynamic, prm.beta, beta_int, rng, cand, cum);
            ++res.evaluations;
            if (cover.profit > it_profit) {
                it_profit = cover.profit;
                it_sel = cover.selected;
            }
        }
        if (it_profit > res.profit) {
            res.profit = it_profit;
            res.solution = it_sel;
            stale = 0;
        } else {
            ++stale;
        }
        const bool global = prm.gb_every > 0 && (t + 1) % prm.gb_every == 0;
        const std::vector<uint8_t>& best = global ? res.solution : it_sel;
        for (int i = 0; i < m; ++i)
            tau[i] = std::clamp((1 - prm.rho) * tau[i] + prm.rho * best[i], prm.tau_min, 1.0);
        if (prm.restart > 0 && stale >= prm.restart) {
            std::fill(tau.begin(), tau.end(), 1.0);
            stale = 0;
        }
        res.history.push_back(res.profit);
    }
    res.elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

}  // namespace aco
