# MMAS parameter tuning

How the tuned MMAS parameters were chosen. Every stage is reproducible from the repo root:
`mmas/tune.sh mmas/tuning/stageK.txt mmas/tuning/stageK RUNS`. Already finished configurations are
skipped, so rerunning only rebuilds `ranking.csv`.

## Protocol

- **Tuning instances** (8 of the 60): `185_200_0.15_0.85`, `200_200_0.10_0.75`, `285_300_0.15_0.85`,
  `300_285_0.10_0.75`, `400_400_0.15_0.85`, `500_485_0.10_0.75` (Set I), plus `600_585_0.10_0.75` and
  `785_800_0.15_0.85` (Set II). They cover m < n, m = n, m > n and both densities. The other 52 instances
  are held out.
- **Budget:** fixed at 20 ants × max(m, n) iterations (`--iter-factor` keeps this when the number of ants changes).
- **Seeds:** the same for every configuration (run r uses seed r).
- **Score:** average over the tuning instances of the gap of the **mean** profit to the best known value
  (`gap_mean_%` in each `ranking.csv`). Best is reported too, but it is noisy.

## Stages

| Stage | Runs | Configurations | Result |
|---|---|---|---|
| 1 | 10 | heuristic {static, dynamic} × β {1,2,3,5,8} × ρ {0.02,…,0.3} × τ_min {0.001,0.01,0.05} = 150 | Dynamic heuristic far better (best 2.24% vs 5.58% static; old default 6.38%); β = 8 is best, at the grid edge |
| 2 | 10 | dynamic, β {8,10,12,15,20} × ρ {0.01,0.02,0.05,0.1} × τ_min {0.01,0.05,0.1,0.2} = 80 | β optimum 10–12, ρ = 0.01 best, τ_min nearly flat |
| 3 | 20 | one factor at a time around β 10, ρ 0.01, τ_min 0.2 (18) | Global-best deposit every 5th iteration: 2.06% → 1.70%. **α = 0 (no pheromone): 2.57%, so the pheromone helps.** α = 1 and 20 ants are best; lower ρ and restarts do not help |
| 4 | 20 | gb_every {1,2,3,5,10} × ρ {0.01,0.02,0.05} × τ_min {0.1,0.2,0.3} = 45 | gb_every 1–2 best (1.53–1.55% on average); top settings 1.35–1.55% |
| 5 | 30 | top 4 of stage 4, β {8,12,15}, restarts {100,250,500} (12) | β 10–15 is a plateau, β 8 worse; restarts do not help; chosen: β 12, ρ 0.01, τ_min 0.2, gb_every 2 |
| 6 | 30 | the 6 smallest instances (85–100 items): β {3,5,8,12} × ρ {0.01,0.02,0.05,0.1} + old default (17) | On small instances **β 3, ρ 0.1** is best (0.81%, BKS on 6/6); β 12 gives 2.4%, the old default 1.94% |

## Tuned MMAS

`--heuristic dynamic --beta 12 --rho 0.01 --tau-min 0.2 --gb-every 2` (α = 1, 20 ants, max(m, n) iterations)

Results with 100 runs on all 60 instances (`mmas/results/mmas_tuned_set{1,2}.csv`), compared with the
old defaults (`--heuristic static --beta 2 --rho 0.1 --tau-min 0.01`, `mmas/results/mmas_set{1,2}.csv`):

| | Gap of best | Gap of mean | BKS reached | Time per run |
|---|---|---|---|---|
| Set I, old defaults | 2.13% | 5.51% | 6/30 | 0.44 s |
| Set I, tuned | 0.84% | 1.72% | 11/30 | 0.80 s |
| Set II, old defaults | 3.33% | 6.66% | 1/30 | 3.23 s |
| Set II, tuned | 0.20% | 1.04% | 19/30 | 4.78 s |
| 52 held-out instances, old defaults | 2.71% | 6.03% | 7/52 | |
| 52 held-out instances, tuned | 0.51% | 1.37% | 26/52 | |

The mean improves on 55 of the 60 instances. The 5 exceptions are the smallest Set I instances (85–100
items, none of them in the tuning set), where stage 6 found β 3, ρ 0.1 better. The best β therefore
depends on the instance size.
